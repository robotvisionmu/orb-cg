"""
Minimal dataset entrypoint for ORB-CG runtime.

Supported datasets:
- replica
- realsense
"""

from orb_cg.dataset_classes import RealsenseDataset, ReplicaDataset
from orb_cg.dataset_helpers import load_dataset_config, measure_time


@measure_time
def get_dataset(dataconfig, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig)
    dataset_name = config_dict["dataset_name"].lower()

    if dataset_name == "replica":
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    if dataset_name == "realsense":
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)

    raise ValueError(
        f"Unsupported dataset name '{config_dict['dataset_name']}'. "
        "This slim build only supports replica and realsense."
    )
