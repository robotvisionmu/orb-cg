import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np

class MappedObject:
    def __init__(self, id, label, pcd_world, bbox, clip_ft):
        self.id = id

        self.label = label
        self.pcd_world = pcd_world
        self.bbox = bbox
        self.clip_ft = clip_ft

        self.contribs = {
            "kfs": [],
            "bboxes_2d": [],
            "class_ids": [],
            "class_labels": [],
            "masks": [],
            "confidences": [],
            "clip_fts": [],
            "pcds_cam": [],
            "pcds_world": [],
            "bboxes_3d": [],
        }

    
    def add_contrib(self, contrib):
        for key in self.contribs.keys():
            self.contribs[key].append(contrib[key])
        
        self.merge_contrib(contrib)

    
    def merge_contrib(self, contrib):
        # self.label = max(self.contribs['class_labels'], key=self.contribs['class_labels'].count)
        self.pcd_world += contrib["pcds_world"]
        self.compute_bounding_box()

        # curr_clip = torch.from_numpy(self.clip_ft)
        # contrib_clip = torch.from_numpy(contrib['clip_fts'])

        # new_clip = (curr_clip * (len(self.contribs['clip_fts']) - 1) + contrib_clip) / len(self.contribs['clip_fts'])
        # new_clip = F.normalize(new_clip, dim=0)

        # self.clip_ft = new_clip.numpy()

    def remove_contrib(self, kf):
        index = self.contribs['kfs'].index(kf)
        
        for key in self.contribs.keys():
            self.contribs[key].pop(index)

        if len(self.contribs['class_labels']) > 0:
            self.label = max(self.contribs['class_labels'], key=self.contribs['class_labels'].count)
        else:
            self.label = "unknown"

        self.update_pcd_world()

        if len(self.contribs['clip_fts']) > 0:
            # Ensure all features are tensors before stacking, then store back as numpy
            clip_tensors = [torch.as_tensor(ft) for ft in self.contribs['clip_fts']]
            clip_mean = torch.stack(clip_tensors, dim=0).mean(dim=0)
            clip_mean = F.normalize(clip_mean, dim=0)
            self.clip_ft = clip_mean.detach().cpu().numpy()
        else:
            # Fallback to a zero vector matching existing dim if available, else default 512
            dim = int(self.clip_ft.shape[-1]) if hasattr(self, 'clip_ft') and hasattr(self.clip_ft, 'shape') else 512
            self.clip_ft = np.zeros((dim,), dtype=np.float32)
    
    def update_pcd_world(self):
        new_pcd = o3d.geometry.PointCloud()

        for contrib_pcd in self.contribs['pcds_world']:
            new_pcd += contrib_pcd

        self.pcd_world = new_pcd
        self.compute_bounding_box()

    def compute_bounding_box(self):
        if len(self.pcd_world.points) >= 4:
            try:
                self.bbox = self.pcd_world.get_oriented_bounding_box(robust=True)
            except RuntimeError as e:
                print(f"Met {e}, use axis aligned bounding box instead")
                self.bbox = self.pcd_world.get_axis_aligned_bounding_box()
        else:
            self.bbox = self.pcd_world.get_axis_aligned_bounding_box()