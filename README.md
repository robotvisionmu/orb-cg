# ORB-CG

**ORB-CG** provides online open-vocabulary object mapping by integrating custom python bindings of ORB-SLAM3 with the core object mapping capability of ConceptGraphs. 

Originally forked from the `ali-dev` branch of [ConceptGraphs](https://github.com/concept-graphs/concept-graphs/tree/ali-dev).  


## Installation
Tested on Ubuntu 24.04 with Python 3.10.

1. Create and activate a conda environment:
```bash
    conda create -n orb-cg python=3.10
    conda activate orb-cg
    conda install matplotlib opencv=4.10 cmake conda-forge::pybind11 conda-forge::libstdcxx-ng numpy=1.26.4
```

2. Install Pangolin:
```bash
    git clone https://github.com/stevenlovegrove/Pangolin.git
    cd Pangolin
    git checkout tags/v0.9.4
    cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
    cmake --build build --target install -j16
    cd ..
```

3. Install pyorbslam3:
```bash
    git clone --recursive https://github.com/robotvisionmu/pyorbslam3.git
    cd pyorbslam3
    pip install .
    cd ..
```

4. Install ORB-CG dependencies:
```bash
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
    conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
    conda install ultralytics open-clip-torch ultralytics-thop -c conda-forge -c dnachun
    pip install "rerun-sdk<0.23" open3d hydra-core openai kornia imageio supervision natsort git+https://github.com/ultralytics/CLIP.git
```

5. Clone and install this repository:
```bash
    git clone https://github.com/robotvisionmu/orb-cg.git
    cd orb-cg
    pip install -e .
```

6. Unzip ORB-SLAM Vocabulary
```bash
    cd orbslam/vocab
    tar -xvzf ORBvoc.txt.tar.gz
    cd ..
```

## Usage
The code is configured to be run on `Replica room0` by default

Download the Replica dataset into `data` directory
```bash
    cd /path/to/orb-cg
    mkdir -p data
    cd data
    wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
    unzip Replica.zip
```

Run ORB-CG:
```bash
    cd /path/to/orb-cg
    python main.py
```