class Detection:
    def __init__(self, keyframe_id, bbox, mask, label, confidence, pcd_cam, pcd_world):
        self.keyframe_id = keyframe_id
        self.object_id = None

        self.bbox = bbox
        self.mask = mask
        self.label = label
        self.confidence = confidence

        self.pcd_cam = pcd_cam
        self.pcd_world = pcd_world

    def update_pcd_world(self, pose):
        self.pcd_world = self.pcd_cam.transform(pose)