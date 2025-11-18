# ORB-CG
ORB-CG provides online open-vocabulary object mapping by integrating ORB-SLAM3 with the object mapping capability of ConceptGraphs.

## Installation
This was tested on Ubuntu 24.04

Create conda environment

    conda create -n orb-cg python=3.10
    conda activate orb-cg
    conda install matplotlib opencv=4.10 cmake conda-forge::pybind11 conda-forge::libstdcxx-ng numpy=1.26.4

Install Pangolin

    git clone https://github.com/stevenlovegrove/Pangolin.git
    git checkout tags/v0.9.4
    cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
    cmake --build build --target install -j16

Install pyorbslam3

    git clone --recursive https://github.com/robotvisionmu/pyorbslam3.git
    cd pyorbslam3
    pip install .

Install CG dependencies
    
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
    conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
    conda install  ultralytics open-clip-torch ultralytics-thop -c conda forge -c dnachun

    pip install "rerun-sdk<0.23" open3d hydra-core openai kornia imageio supervision natsort git+https://github.com/ultralytics/CLIP.git


Clone this repository

    git clone --recursive https://github.com/robotvisionmu/orb-cg.git
    cd orb-cg/concept-graphs
    pip install . -e

Unzip ORB-SLAM vocabulary

    cd orb-cg/orbslam_files
    tar -xvzf ORBvoc.txt.tar.gz

Set `repo_root` and `data_root` in `orb-cg/concept-graphs/conceptgraph/hydra_configs/base_paths.yaml`

    repo_root: /home/<user>/<code-directory>/orb-cg/concept-graphs
    data_root: <path-to-data>/data


## Usage
The code is configured to be run on Replica room0 by default

Download the Replica dataset to your `data_root` directory

    cd <data_root>
    wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
    unzip Replica.zip

Run `orb-cg.py` to perform online object mapping and visualise in a rerun viewer

    cd <repo_root>
    python conceptgraph/slam/orb-cg.py
