# Standard library
import logging
import os
from pathlib import Path

# Third-party
import cv2
import hydra
import numpy as np
import open_clip
import orbslam3
import torch
from omegaconf import DictConfig
from tqdm import trange
from ultralytics import SAM, YOLO, YOLOWorld

# Local project
from orb_cg.datasets_common import get_dataset
from orb_cg.key_frame import KeyFrame
from orb_cg.mapped_object import MappedObject
from orb_cg.rerun_wrapper import (
    ReRunWrapper,
    orr_log_objs_pcd_and_bbox,
    rr_log_camera,
    rr_log_depth,
    rr_log_image,
    rr_log_trajectory,
)
from orb_cg.utils import (
    ClassCatalog,
    aggregate_similarities,
    cfg_to_dict,
    compute_3d_bboxes_from_pcds,
    compute_clip_features,
    compute_detections_pcds_camera,
    compute_detections_pcds_world,
    compute_spatial_similarities,
    compute_visual_similarities,
    filter_detections,
    match_detections_to_objects,
    process_cfg,
    process_pcd,
    require_pose_update,
    resize_detections_torch,
    subtract_contained_masks,
)

# Suppress PIL warnings
logging.getLogger("PIL").setLevel(logging.WARNING)

# Disable torch gradient computation
torch.set_grad_enabled(False)

@hydra.main(version_base=None, config_path="configs/hydra/", config_name="orb-cg")
def main(cfg : DictConfig):
    count = 0
    orr = ReRunWrapper()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("ORB-CG: Online Open-vocab Object Mapping")
    orr.spawn()

    cfg = process_cfg(cfg)

    dataset_config_path = cfg.dataset_config

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=dataset_config_path,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float,
    )

    detections_exp_cfg = cfg_to_dict(cfg)

    obj_classes = ClassCatalog(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    print("\n".join(["Running detections..."] * 3))

    # Initialise Object Detection Model
    # obj_det_model = YOLOWorld('yolov8l-worldv2.pt')
    detection_model_path = cfg.model_paths.detection
    obj_det_model = YOLO(detection_model_path)
    obj_det_model.set_classes(obj_classes.get_classes_arr())

    # Initialize Segmentation Model
    # seg_model = SAM('sam2.1_l.pt')
    segmentation_model_path = cfg.model_paths.segmentation
    seg_model = SAM(segmentation_model_path)

    # Initialise Embedding Model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
    clip_model = clip_model.to(cfg.device)

    # Initialise SLAM System
    vocab_file = cfg.slam_vocab_path
    settings_file = cfg.slam_settings_path
    output_path = cfg.output_root
    rgb_path = os.path.join(output_path, "rgb")
    d_path = os.path.join(output_path, "depth")
    yolo_path = os.path.join(output_path, "yolo")
    sam_path = os.path.join(output_path, "sam")
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(d_path, exist_ok=True)
    os.makedirs(yolo_path, exist_ok=True)
    os.makedirs(sam_path, exist_ok=True)
    
    slam = orbslam3.system(vocab_file, settings_file, orbslam3.Sensor.RGBD)
    slam.set_use_viewer(False)
    slam.initialize()
    
    kf_database = {}
    object_database = {}   

    live_poses = [] 
    processing_times = []

    for frame_idx in trange(len(dataset)):

        # Load RGB-D image
        color_path = dataset.color_paths[frame_idx]
        depth_path = dataset.depth_paths[frame_idx]
        image_orb = cv2.imread(color_path)
        depth_orb = cv2.imread(depth_path, -1)

        slam.process_image_rgbd(image_orb, depth_orb, float(frame_idx))
        keyframe_ids, keyframe_poses, keyframe_maps = slam.get_all_keyframe_data()
        # keyframe_ids = slam.get_all_keyframe_times()
        # keyframe_poses = slam.get_all_keyframe_poses()

        # Determine new and removed keyframes
        prev_keyframe_ids = set(kf_database.keys())
        current_keyframe_ids = set(keyframe_ids)
        new_keyframe_ids = current_keyframe_ids - prev_keyframe_ids
        removed_keyframe_ids = prev_keyframe_ids - current_keyframe_ids

        # print("New Key Frames:", new_keyframe_ids)
        # print("Removed Key Frames:", removed_keyframe_ids)


        updated_object_ids = set()
        # Process removed keyframes
        for kf_id in removed_keyframe_ids:
            for obj_id in list(kf_database[kf_id].obj_associations):
                obj = object_database.get(obj_id)
                if obj is None:
                    continue
                if len(obj.contribs['kfs']) == 1:
                    # Remove object entirely and clean up stale associations across other keyframes
                    del object_database[obj_id]
                    for kf_other in kf_database.values():
                        if obj_id in kf_other.obj_associations:
                            kf_other.obj_associations.remove(obj_id)
                else:
                    object_database[obj_id].remove_contrib(kf_database[kf_id])
                    updated_object_ids.add(obj_id)
            del kf_database[kf_id]
            
        # Process updated poses
        for kf_id in (prev_keyframe_ids - removed_keyframe_ids):
            kf = kf_database[kf_id]

            prev_pose = kf.pose
            curr_pose = keyframe_poses[keyframe_ids.index(kf_id)]

            if require_pose_update(prev_pose, curr_pose):
                kf.pose = curr_pose
                updated_object_ids.update(kf.obj_associations)

        # Update the PCD of all affected objects
        for obj_id in list(updated_object_ids):
            if obj_id in object_database:
                object_database[obj_id].update_pcd_world()


        # Process new keyframes
        for kf_id in new_keyframe_ids:
            live_poses.append(keyframe_poses[keyframe_ids.index(kf_id)])

            frame_idx = int(kf_id)
            color_path = Path(dataset.color_paths[frame_idx])
            color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]
            depth_tensor = depth_tensor[..., 0]
            depth = depth_tensor.cpu().numpy()

            color_np = color_tensor.cpu().numpy() # (H, W, 3)
            image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
            assert image_rgb.max() > 1, "Image is not in range [0, 255]"
            
            image = cv2.imread(str(color_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create new keyframe
            kf = KeyFrame(kf_id, keyframe_poses[keyframe_ids.index(kf_id)])
            kf_database[kf_id] = kf

            orr.set_time_sequence("frame", frame_idx)

            # Object Detection
            obj_det_results = obj_det_model.predict(image_rgb, conf=0.1, verbose=False)
            bboxes_tensor = obj_det_results[0].boxes.xyxy
            bboxes_np = bboxes_tensor.cpu().numpy()
            confidences = obj_det_results[0].boxes.conf.cpu().numpy()
            class_ids = obj_det_results[0].boxes.cls.cpu().numpy().astype(int)
            class_labels = [obj_det_results[0].names[class_id] for class_id in class_ids]

            annotated_image = obj_det_results[0].plot()

            # Segmentation
            if len(obj_det_results[0].boxes) != 0:
                seg_results = seg_model.predict(color_path, bboxes=bboxes_tensor, verbose=False)
                masks = seg_results[0].masks.data.cpu().numpy()
            else:
                masks = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

            seg_annotated_image = seg_results[0].plot()

            detections = {
                "bboxes_2d": bboxes_np,
                "class_ids": class_ids,
                "class_labels": class_labels,
                "masks": masks,
                "confidences": confidences,
            }

            # Save outputs
            depth_uint8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

            # Apply a color map (use CV_8UC1 format)
            colored_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(rgb_path, f"{frame_idx:06d}.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(d_path, f"{frame_idx:06d}.png"), colored_depth)
            cv2.imwrite(os.path.join(yolo_path, f"{frame_idx:06d}_yolo.png"), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(sam_path, f"{frame_idx:06d}_sam.png"), cv2.cvtColor(seg_annotated_image, cv2.COLOR_RGB2BGR))
            
            # Compute CLIP features for each detection
            clip_fts = compute_clip_features(image_rgb, detections, clip_model, clip_preprocess, cfg.device)
            
            # Resize masks and bboxes to match the image size
            detections = resize_detections_torch(detections, image_rgb)

            # Add Image Features to detections
            detections["clip_fts"] = clip_fts

            # filter the observations
            detections = filter_detections(detections, image_rgb,
                skip_bg=cfg.skip_bg,
                BG_CLASSES=obj_classes.get_bg_classes_arr(),
                mask_area_threshold=cfg.mask_area_threshold,
                max_bbox_area_ratio=cfg.max_bbox_area_ratio,
                mask_conf_threshold=cfg.mask_conf_threshold,
            )

            # Skip to next frame if no detections
            if len(detections['masks']) == 0: 
                continue

            # Separate small objects that are contained within larger objects (e.g., pillows on couches)
            detections['masks'] = subtract_contained_masks(detections['bboxes_2d'], detections['masks'])

            # Compute 3D point cloud for each detection
            detections_pcds_cam = compute_detections_pcds_camera(
                depth=depth,
                masks=detections['masks'],
                cam_K=intrinsics.cpu().numpy()[:3, :3],
                image_rgb=image_rgb,
                min_points_threshold=cfg.min_points_threshold,
                obj_pcd_max_points=cfg.obj_pcd_max_points,
                device=cfg.device,
            )

            detections['pcds_cam'] = detections_pcds_cam

            # Some detections may not have valid PCDs (e.g. too few points). Remove these detections
            remove_indices = [i for i, pcd in enumerate(detections_pcds_cam) if pcd == None]
            for key, value in detections.items():
                if isinstance(value, list):
                    detections[key] = [v for i, v in enumerate(value) if i not in remove_indices]
                elif isinstance(value, np.ndarray):
                    detections[key] = np.delete(value, remove_indices, axis=0)

            # pose = dataset.poses[frame_idx]
            pose = keyframe_poses[keyframe_ids.index(kf_id)]
            # pose = poses[frame_idx]
            detections_pcds_world = compute_detections_pcds_world(detections['pcds_cam'], pose)
            detections['pcds_world'] = detections_pcds_world

            for i, pcd in enumerate(detections['pcds_world']):
                detections['pcds_world'][i] = process_pcd(
                    pcd,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )

            # Compute 3D bounding boxes for each detection
            detections_3d_bboxes_world = compute_3d_bboxes_from_pcds(detections['pcds_world'])
            detections['bboxes_3d'] = detections_3d_bboxes_world

            # If Object database is empty, add all detections as new objects
            if len(object_database) == 0:
                for det_idx in range(len(detections['masks'])):
                    print("First frame: Adding all detected objects to empty database")
                    # obj_id = str(uuid.uuid4())
                    obj_id = count
                    count += 1
                    new_obj = MappedObject(obj_id, detections['class_labels'][det_idx], detections['pcds_world'][det_idx], detections['bboxes_3d'][det_idx], detections['clip_fts'][det_idx])
                    
                    
                    contrib = {k: v[det_idx] for k, v in detections.items()}
                    contrib['kfs'] = kf
                    new_obj.add_contrib(contrib)
                    object_database[obj_id] = new_obj
                    kf.obj_associations.append(obj_id)
            else:
                # Compute similarities
                spatial_sim = compute_spatial_similarities(detections=detections, object_database=object_database, downsample_voxel_size=cfg['downsample_voxel_size'])
                visual_sim = compute_visual_similarities(detections=detections, object_database=object_database)
                agg_sim = aggregate_similarities(match_method=cfg['match_method'], phys_bias=cfg['phys_bias'], spatial_sim=spatial_sim, visual_sim=visual_sim)

                # Match detections to existing objects
                match_indices = match_detections_to_objects(agg_sim=agg_sim, detection_threshold=cfg['sim_threshold'])
   
                # Now merge the detected objects into the existing objects based on the match indices
                for det_idx, obj_idx in enumerate(match_indices):
                    if obj_idx == None:
                        # Create new object
                        # obj_id = str(uuid.uuid4())
                        obj_id = count
                        count += 1
                        new_obj = MappedObject(obj_id, detections['class_labels'][det_idx], detections['pcds_world'][det_idx], detections['bboxes_3d'][det_idx], detections['clip_fts'][det_idx])
                        contrib = {k: v[det_idx] for k, v in detections.items()}
                        contrib['kfs'] = kf
                        new_obj.add_contrib(contrib)
                        object_database[obj_id] = new_obj
                        kf.obj_associations.append(obj_id)
                        updated_object_ids.add(obj_id)
                    else:
                        # Merge with existing object
                        obj = list(object_database.values())[obj_idx]
                        contrib = {k: v[det_idx] for k, v in detections.items()}
                        contrib['kfs'] = kf
                        obj.add_contrib(contrib)
                        kf.obj_associations.append(obj.id)
                        updated_object_ids.add(obj.id)
            
            rr_log_camera(live_poses[-1], color_path, intrinsics, cfg.image_width, cfg.image_height, frame_idx)
            rr_log_image(annotated_image, frame_idx)
            rr_log_depth(depth, frame_idx)
        
            if len(live_poses) > 1:
                rr_log_trajectory(live_poses[-1], live_poses[-2], frame_idx, "trajectory_live")
            
        # TODO: Only re-log objects that have been updated in this frame
        # orr_log_objs_pcd_and_bbox(object_database, obj_classes)
        if updated_object_ids:
            objs_to_log = {oid: object_database[oid] for oid in updated_object_ids if oid in object_database}
            orr_log_objs_pcd_and_bbox(objs_to_log, obj_classes)

        # TODO: Only re-log if poses have been updated
        for i in range(len(keyframe_poses)):
            rr_log_trajectory(keyframe_poses[i], keyframe_poses[i-1], keyframe_ids[i], "trajectory_optimised")

    slam.shutdown()
    while not slam.is_shutdown():
        continue

    # TODO: Any processing after SLAM shutdown?

if __name__ == "__main__":
    main()
