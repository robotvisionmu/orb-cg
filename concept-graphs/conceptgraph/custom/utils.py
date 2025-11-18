import copy
import cv2
import torch
import faiss
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from PIL import Image
from collections import Counter

def require_pose_update(old_pose, new_pose, trans_thresh_m=0.01, rot_thresh_deg=1.0):
    if np.allclose(old_pose, new_pose):
        return False

    trans_delta = np.linalg.norm(new_pose[:3, 3] - old_pose[:3, 3])

    R1 = old_pose[:3, :3]
    R2 = new_pose[:3, :3]
    R_delta = R1.T @ R2

    angle_delta = np.degrees(np.arccos((np.trace(R_delta)-1)/2))

    return trans_delta > trans_thresh_m or angle_delta > rot_thresh_deg

def compute_clip_features(image, detections, clip_model, clip_preprocess, device):
    image = Image.fromarray(image)
    padding = 20
    
    preprocessed_images = []

    # Prepare data for batch processing
    for idx in range(len(detections['bboxes_2d'])):
        x_min, y_min, x_max, y_max = detections['bboxes_2d'][idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

    # Convert lists to batches
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)
    
    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    
    return image_feats

def resize_detections(detections, image):

    # If the shapes are the same, no resizing is necessary
    if detections['masks'].shape[1:] == image.shape[:2]:
        return detections

    resized_masks = []

    for mask_idx in range(len(detections['masks'])):
        mask = detections['masks'][mask_idx]
        
        # Reshape the mask to the image shape
        mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        resized_masks.append(mask)

        # Rescale the bboxes coordinates to the image shape
        x1, y1, x2, y2 = detections['bboxes'][mask_idx]
        x1 = round(x1 * image.shape[1] / mask.shape[1])
        y1 = round(y1 * image.shape[0] / mask.shape[0])
        x2 = round(x2 * image.shape[1] / mask.shape[1])
        y2 = round(y2 * image.shape[0] / mask.shape[0])
        detections['bboxes'][mask_idx] = np.array([x1, y1, x2, y2])

    if len(resized_masks) > 0:
        detections['masks'] = np.asarray(resized_masks)

    return detections

def resize_detections_torch(detections, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print(type(detections['masks']))
    # print(detections['masks'].shape)
    
    masks = torch.tensor(detections['masks'], dtype=torch.float32, device=device)  # Shape: (N, H_mask, W_mask)
    bboxes = torch.tensor(detections['bboxes_2d'], dtype=torch.float32, device=device)  # Shape: (N, 4)

    image_h, image_w = image.shape[0], image.shape[1]
    mask_h, mask_w = masks.shape[1], masks.shape[2]

    # If masks already match original image size, return early
    if mask_h == image_h and mask_w == image_w:
        return detections

    # Rescale bounding boxes coordinates (vectorized)
    scale_x = image_w / mask_w
    scale_y = image_h / mask_h

    # bboxes = (xmin, ymin, xmax, ymax)
    bboxes[:, [0, 2]] = torch.round(bboxes[:, [0, 2]] * scale_x)
    bboxes[:, [1, 3]] = torch.round(bboxes[:, [1, 3]] * scale_y)

    # Resize masks in batch to original image size using nearest interpolation
    masks = masks.unsqueeze(1)  # Add channel dim: (N, 1, H_mask, W_mask)
    resized_masks = F.interpolate(masks, size=(image_h, image_w), mode='nearest')
    resized_masks = resized_masks.squeeze(1)  # Back to shape (N, image_h, image_w)

    # Convert masks back to bool numpy arrays
    detections['masks'] = resized_masks.cpu().numpy().astype(bool)
    detections['bboxes'] = bboxes.cpu().numpy().astype(int)

    return detections

def filter_detections(detections: dict, image: np.ndarray, skip_bg: bool = None, BG_CLASSES: list = None, mask_area_threshold: float = 10, max_bbox_area_ratio: float = None, mask_conf_threshold: float = None):
    # No detections in this frame
    if len(detections['bboxes_2d']) == 0:
        return detections

    # Filter out the objects based on various criteria
    detections_to_keep = []
    for mask_idx in range(len(detections['bboxes_2d'])):
        class_label = detections['class_labels'][mask_idx]

        # Remove detection masks that are too small
        mask_area = detections['masks'][mask_idx].sum()
        if mask_area < max(mask_area_threshold, 10):
            continue

        # Remove background detections if specified
        if skip_bg and class_label in BG_CLASSES:
            continue

        # Remove non-background detection bboxes that are too large
        if class_label not in BG_CLASSES:
            x1, y1, x2, y2 = detections['bboxes_2d'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if max_bbox_area_ratio is not None and bbox_area > max_bbox_area_ratio * image_area:
                continue

        # Remove detections with low confidence
        if mask_conf_threshold is not None and detections['confidences'] is not None:
            if detections['confidences'][mask_idx] < mask_conf_threshold:
                continue

        detections_to_keep.append(mask_idx)

    for key, value in detections.items():
        if isinstance(value, np.ndarray):
            detections[key] = value[detections_to_keep]
        else:
            detections[key] = [value[i] for i in detections_to_keep]

    return detections

def subtract_contained_masks(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    '''
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.
     
    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2
        
    Returns:
        net_mask: (N, H, W), binary mask
    '''

    # Compute areas of all boxes
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])  # right-bottom points (N, N, 2)
    inter = (rb - lt).clip(min=0)  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1] # (N, N)
    
    # Compute IoU
    inter_ratio_over_box1 = inter_areas / areas[:, None] # (N, N)
    inter_ratio_over_box2 = inter_ratio_over_box1.T # (N, N)

    # If more than th1 of box2 is contained by box1, and less than th2 of box1 is covered by box2, then box2 is contained by box1
    contained = (inter_ratio_over_box1 < th2) & (inter_ratio_over_box2 > th1) # (N, N)
    contained_idx = contained.nonzero() # (num_contained, 2) indices of contained boxes

    net_mask = mask.copy() # (N, H, W)

    for i in range(len(contained_idx[0])):
        net_mask[contained_idx[0][i]] = net_mask[contained_idx[0][i]] & (~net_mask[contained_idx[1][i]])

    return net_mask


def compute_detections_pcds_camera(
    depth, 
    masks, 
    cam_K, 
    image_rgb, 
    min_points_threshold=5,
    obj_pcd_max_points = None,
    device='cuda'
):
    """
    This function processes a batch of objects to create colored point clouds, apply transformations, and compute bounding boxes.

    Args:
        depth_array (numpy.ndarray): Array containing depth values.
        masks (numpy.ndarray): Array containing binary masks for each object.
        cam_K (numpy.ndarray): Camera intrinsic matrix.
        image_rgb (numpy.ndarray, optional): RGB image. Defaults to None.
        trans_pose (numpy.ndarray, optional): Transformation matrix. Defaults to None.
        min_points_threshold (int, optional): Minimum number of points required for an object. Defaults to 5.
        spatial_sim_type (str, optional): Type of spatial similarity. Defaults to 'axis_aligned'.
        device (str, optional): Device to use. Defaults to 'cuda'.

    Returns:
        list: List of dictionaries containing processed objects. Each dictionary contains a point cloud and a bounding box.
    """
    N, H, W = masks.shape

    # Convert inputs to tensors and move to the specified device
    depth_tensor = torch.from_numpy(depth).to(device).float()
    masks_tensor = torch.from_numpy(masks).to(device).float()
    cam_K_tensor = torch.from_numpy(cam_K).to(device).float()
    rgb_tensor = torch.from_numpy(image_rgb).to(device).float() / 255.0  # Normalize RGB values

    points_tensor, colors_tensor = detection_to_3D(depth_tensor, masks_tensor, cam_K_tensor, rgb_tensor, device)

    pcds_cam = [None] * N
    # pcds_world = [None] * N
    
    for i in range(N):
        mask_points = points_tensor[i]
        mask_colors = colors_tensor[i]

        valid_points_mask = mask_points[:, :, 2] > 0
        if torch.sum(valid_points_mask) < min_points_threshold:
            continue

        valid_points = mask_points[valid_points_mask]
        valid_colors = mask_colors[valid_points_mask]

        downsampled_points, downsampled_colors = dynamic_downsample(valid_points, colors=valid_colors, target=obj_pcd_max_points)

        ds_points = downsampled_points.cpu().numpy()
        ds_colors = downsampled_colors.cpu().numpy()
    
        # Create Open3D point cloud
        pcd_cam = o3d.geometry.PointCloud()
        pcd_cam.points = o3d.utility.Vector3dVector(ds_points)
        pcd_cam.colors = o3d.utility.Vector3dVector(ds_colors)

        # bbox = None
        # if len(pcd_cam.points) >= 4:
        #     try:
        #         bbox = pcd_cam.get_oriented_bounding_box(robust=True)
        #     except RuntimeError as e:
        #         print(f"Met {e}, use axis aligned bounding box instead")
        #     bbox = pcd_cam.get_axis_aligned_bounding_box()
        # else:
        #     bbox = pcd_cam.get_axis_aligned_bounding_box()

        # if bbox.volume() < 1e-6:
        #     continue

        pcds_cam[i] = pcd_cam

    return pcds_cam

def detection_to_3D(
    depth_tensor: torch.Tensor,
    masks_tensor: torch.Tensor,
    cam_K: torch.Tensor,
    image_rgb_tensor: torch.Tensor,
    device: str = 'cuda'
) -> tuple:
    """
    Converts a batch of masked depth images to 3D points and corresponding colors.

    Args:
        depth_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the depth images.
        masks_tensor (torch.Tensor): A tensor of shape (N, H, W) representing the masks for each depth image.
        cam_K (torch.Tensor): A tensor of shape (3, 3) representing the camera intrinsic matrix.
        image_rgb_tensor (torch.Tensor, optional): A tensor of shape (N, H, W, 3) representing the RGB images. Defaults to None.
        device (str, optional): The device to perform the computation on. Defaults to 'cuda'.

    Returns:
        tuple: A tuple containing the 3D points tensor of shape (N, H, W, 3) and the colors tensor of shape (N, H, W, 3).
    """
    # Mask dimensions
    N, H, W = masks_tensor.shape

    # Camera intrinsics
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    
    # Generate grid of pixel coordinates
    y, x = torch.meshgrid(torch.arange(0, H, device=device), torch.arange(0, W, device=device), indexing='ij')
    
    # Apply masks to depth
    z = depth_tensor.repeat(N, 1, 1) * masks_tensor

    # Valid mask for points with positive depth
    valid = (z > 0).float()

    # Compute 3D coordinates
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    points = torch.stack((x, y, z), dim=-1) * valid.unsqueeze(-1)  # Shape: (N, H, W, 3)

    # Repeat RGB image for each mask and apply masks
    repeated_rgb = image_rgb_tensor.repeat(N, 1, 1, 1) * masks_tensor.unsqueeze(-1)
    colors = repeated_rgb * valid.unsqueeze(-1)

    return points, colors

def dynamic_downsample(points, colors=None, target=5000):
    """
    Simplified and configurable downsampling function that dynamically adjusts the 
    downsampling rate based on the number of input points. If a target of -1 is provided, 
    downsampling is bypassed, returning the original points and colors.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) for N points.
        target (int): Target number of points to aim for in the downsampled output, 
                      or -1 to bypass downsampling.
        colors (torch.Tensor, optional): Corresponding colors tensor of shape (N, 3). 
                                         Defaults to None.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Downsampled points and optionally 
                                                     downsampled colors, or the original 
                                                     points and colors if target is -1.
    """
    # Check if downsampling is bypassed
    if target == -1:
        return points, colors
    
    num_points = points.size(0)
    
    # If the number of points is less than or equal to the target, return the original points and colors
    if num_points <= target:
        return points, colors
    
    # Calculate downsampling factor to aim for the target number of points
    downsample_factor = max(1, num_points // target)
    
    # Select points based on the calculated downsampling factor
    downsampled_points = points[::downsample_factor]
    
    # If colors are provided, downsample them with the same factor
    downsampled_colors = colors[::downsample_factor] if colors is not None else None

    return downsampled_points, downsampled_colors

def compute_detections_pcds_world(detections_pcds_cam, transform):
    pcds_world = []
    for pcd_cam in detections_pcds_cam:
        pcd_world = copy.deepcopy(pcd_cam)
        pcd_world.transform(transform)
        pcds_world.append(pcd_world)
    return pcds_world

def compute_3d_bboxes_from_pcds(pcds_cam,):
    bboxes = []
    for pcd in pcds_cam:
        if len(pcd.points) >= 4:
            try:
                bbox = pcd.get_oriented_bounding_box(robust=True)
            except RuntimeError as e:
                print(f"Met {e}, use axis aligned bounding box instead")
            bbox = pcd.get_axis_aligned_bounding_box()
        else:
            bbox = pcd.get_axis_aligned_bounding_box()

        bboxes.append(bbox)

    return bboxes   

def compute_visual_similarities(detections, object_database) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    det_fts = []
    obj_fts = []

    for det_ft in detections['clip_fts']:
        # Support either numpy arrays or tensors
        det_fts.append(torch.as_tensor(det_ft))
    det_fts = torch.stack(det_fts, dim=0) # (M,

    for obj in object_database.values():
        # Support either numpy arrays or tensors
        obj_fts.append(torch.as_tensor(obj.clip_ft))
    obj_fts = torch.stack(obj_fts, dim=0) # (N, D)

    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    
    return visual_sim

def compute_spatial_similarities(detections, object_database, downsample_voxel_size) -> torch.Tensor:
    spatial_sim = compute_overlap_matrix_general(object_database, detections, downsample_voxel_size)
    spatial_sim = torch.from_numpy(spatial_sim).T
    
    return spatial_sim

def aggregate_similarities(match_method: str, phys_bias: float, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if match_method == "sim_sum":
        sims = (1 + phys_bias) * spatial_sim + (1 - phys_bias) * visual_sim
    else:
        raise ValueError(f"Unknown matching method: {match_method}")
    
    return sims

def compute_overlap_matrix_general(object_database, detections, downsample_voxel_size = None) -> np.ndarray:
    """
    Compute the overlap matrix between two sets of objects represented by their point clouds. This function can also perform self-comparison when `objects_b` is not provided. The overlap is quantified based on the proximity of points from one object to the nearest points of another, within a threshold specified by `downsample_voxel_size`.

    Parameters
    ----------
    objects_a : MapObjectList
        A list of object representations where each object contains a point cloud ('pcd') and bounding box ('bbox').
        This is the primary set of objects for comparison.

    objects_b : Optional[MapObjectList]
        A second list of object representations similar to `objects_a`. If None, `objects_a` will be compared with itself to calculate self-overlap. Defaults to None.

    downsample_voxel_size : Optional[float]
        The threshold for determining whether points are close enough to be considered overlapping. Specifically, it's the square of the maximum distance allowed between points from two objects to consider those points as overlapping.
        Must be provided; if None, a ValueError is raised.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (len(objects_a), len(objects_b)) containing the overlap ratios between objects.
        The overlap ratio is defined as the fraction of points in the second object's point cloud that are within `downsample_voxel_size` distance to any point in the first object's point cloud.

    Raises
    ------
    ValueError
        If `downsample_voxel_size` is not provided.

    Notes
    -----
    The function uses the FAISS library for efficient nearest neighbor searches to compute the overlap.
    Additionally, it employs a 3D IoU (Intersection over Union) computation for bounding boxes to quickly filter out pairs of objects without spatial overlap, improving performance.
    - The overlap matrix helps identify potential duplicates or matches between new and existing objects based on spatial overlap.
    - High values (e.g., >0.8) in the matrix suggest a significant overlap, potentially indicating duplicates or very close matches.
    - Moderate values (e.g., 0.5-0.8) may indicate similar objects with partial overlap.
    - Low values (<0.5) generally suggest distinct objects with minimal overlap.
    - The choice of a "match" threshold depends on the application's requirements and may require adjusting based on observed outcomes.

    Examples
    --------
    >>> objects_a = [{'pcd': pcd1, 'bbox': bbox1}, {'pcd': pcd2, 'bbox': bbox2}]
    >>> objects_b = [{'pcd': pcd3, 'bbox': bbox3}, {'pcd': pcd4, 'bbox': bbox4}]
    >>> downsample_voxel_size = 0.05
    >>> overlap_matrix = compute_overlap_matrix_general(objects_a, objects_b, downsample_voxel_size)
    >>> print(overlap_matrix)
    """
    # if downsample_voxel_size is None, raise an error
    if downsample_voxel_size is None:
        raise ValueError("downsample_voxel_size is not provided")

    # hardcoding for now because its this value is actually not supposed to be the downsample voxel size
    downsample_voxel_size = 0.025

    len_a = len(object_database)
    len_b = len(detections['masks'])
    overlap_matrix = np.zeros((len_a, len_b))

    # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
    points_a = [np.asarray(obj.pcd_world.points, dtype=np.float32) for obj in object_database.values()] # m arrays
    indices_a = [faiss.IndexFlatL2(points_a_arr.shape[1]) for points_a_arr in points_a] # m indices

    # Add the points from the numpy arrays to the corresponding FAISS indices
    for idx_a, points_a_arr in zip(indices_a, points_a):
        idx_a.add(points_a_arr)

    points_b = [np.asarray(pcd.points, dtype=np.float32) for pcd in detections['pcds_world']] # n arrays

    obj_bboxes = []
    for obj in object_database.values():
        obj_bboxes.append(torch.from_numpy(np.asarray(obj.bbox.get_box_points())))
    bbox_a = torch.stack(obj_bboxes, dim=0)
    
    det_bboxes = []
    for bbox in detections['bboxes_3d']:
        det_bboxes.append(torch.from_numpy(np.asarray(bbox.get_box_points())))
    bbox_b = torch.stack(det_bboxes, dim=0)
    
    ious = compute_3d_iou_accurate_batch(bbox_a, bbox_b) # (m, n)

    # Compute the pairwise overlaps
    for idx_a in range(len_a):
        for idx_b in range(len_b):
            # skip if the boxes do not overlap at all
            if ious[idx_a,idx_b] < 1e-6:
                continue

            # get the distance of the nearest neighbor of
            # each point in points_b[idx_b] to the points_a[idx_a]
            D, I = indices_a[idx_a].search(points_b[idx_b], 1) 
            overlap = (D < downsample_voxel_size ** 2).sum() # D is the squared distance

            # Calculate the ratio of points within the threshold
            overlap_matrix[idx_a, idx_b] = overlap / len(points_b[idx_b])

    return overlap_matrix

def compute_3d_iou_accurate_batch(bbox1, bbox2):
    '''
    Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.
    
    bbox1: (M, 8, D), e.g. (M, 8, 3)
    bbox2: (N, 8, D), e.g. (N, 8, 3)
    
    returns: (M, N)
    '''
    # Must expend the box beforehand, otherwise it may results overestimated results
    bbox1 = expand_3d_box(bbox1, 0.02)
    bbox2 = expand_3d_box(bbox2, 0.02)
    
    import pytorch3d.ops as ops

    bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]
    
    inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())
    
    return iou

def expand_3d_box(bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
    '''
    Expand the side of 3D boxes such that each side has at least eps length.
    Assumes the bbox cornder order in open3d convention. 
    
    bbox: (N, 8, D)
    
    returns: (N, 8, D)
    '''
    center = bbox.mean(dim=1)  # shape: (N, D)

    va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
    vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
    vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)
    
    a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
    
    va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
    vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
    vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)
    
    new_bbox = torch.stack([
        center - va/2.0 - vb/2.0 - vc/2.0,
        center + va/2.0 - vb/2.0 - vc/2.0,
        center - va/2.0 + vb/2.0 - vc/2.0,
        center - va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 + vc/2.0,
        center - va/2.0 + vb/2.0 + vc/2.0,
        center + va/2.0 - vb/2.0 + vc/2.0,
        center + va/2.0 + vb/2.0 - vc/2.0,
    ], dim=1) # shape: (N, 8, D)
    
    new_bbox = new_bbox.to(bbox.device)
    new_bbox = new_bbox.type(bbox.dtype)
    
    return new_bbox

def aggregate_similarities(match_method: str, phys_bias: float, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if match_method == "sim_sum":
        sims = (1 + phys_bias) * spatial_sim + (1 - phys_bias) * visual_sim
    else:
        raise ValueError(f"Unknown matching method: {match_method}")
    
    return sims


def match_detections_to_objects(agg_sim: torch.Tensor, detection_threshold: float = float('-inf')):
    """
    Matches detections to objects based on similarity, returning match indices or None for unmatched.

    Args:
        agg_sim: Similarity matrix (detections vs. objects).
        detection_threshold: Threshold for a valid match (default: -inf).

    Returns:
        List of matching object indices (or None if unmatched) for each detection.
    """
    match_indices = []
    for detected_obj_idx in range(agg_sim.shape[0]):
        max_sim_value = agg_sim[detected_obj_idx].max()
        if max_sim_value <= detection_threshold:
            match_indices.append(None)
        else:
            match_indices.append(agg_sim[detected_obj_idx].argmax().item())

    return match_indices

def process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points, run_dbscan=True):
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    
    if dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(
            pcd, 
            eps=dbscan_eps, 
            min_points=dbscan_min_points
        )
        
    return pcd

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
    # Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        
    return pcd