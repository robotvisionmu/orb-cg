import copy

class KeyFrame:
    def __init__(self, id, pose):
        self.id = id
        self.pose = pose
        self.obj_associations = []