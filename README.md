# FlyMeThrough: Human-AI Collaborative 3D Indoor Mapping with Commodity Drones

**[Makeability Lab](https://makeabilitylab.cs.washington.edu/)**

[*Xia Su*](https://xiasu.github.io/) *, [*Ruiqi Chen*]((https://ruiqi-chen-0216.github.io/)) *, [Jingwei Ma](https://jingweim.github.io/), [Chu Li](https://www.chu-li.me/), [Jon E. Froehlich](https://jonfroehlich.github.io/) 

( **<sup>*</sup>** means Equal Contribution)

[[`Paper is coming soon`]()] [[`Project`]()] [[`Demo`]()] 

This repo accompanies our UIST 2025 paper:

> **FlyMeThrough: Human-AI Collaborative 3D Indoor Mapping with Commodity Drones**
> *Xia Su, Ruiqi Chen, Jingwei Ma, Chu Li, Jon E. Froehlich*
> In *Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology (UIST ’25)*
> Busan, Republic of Korea, September 28–October 1, 2025.
> DOI: [10.1145/XXXXX](https://doi.org/10.1145/XXXXXX)

<!--![SAM 2 architecture](assets/model_diagram.png?raw=true) -->

**FlyMeThrough** is a description here



## Installation

This project depends on two external modules: [**SAM2**](https://github.com/facebookresearch/sam2) and [**Depth-Pro**](https://github.com/apple/ml-depth-pro). Please make sure both are properly installed **before running this code**.

We recommend creating a single virtual environment named `flymethrough` and installing both **SAM2** and **Depth-Pro** inside it. Using miniconda:

```bash
conda create -n flymethrough -y python=3.10
conda activate flymethrough
```

The code requires `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

cd ..  # go back to the project root

git clone https://github.com/apple/ml-depth-pro.git && cd ml-depth-pro
pip install -e .
cd ..  # go back to the project root
```

If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

After installing **SAM2** and **Depth-Pro**, you also need to install the dependencies specific to this project.  


```bash
pip install -r requirements.txt
```

Note:
1. It's recommended to create a new Python environment via [Anaconda](https://www.anaconda.com/) for this installation and install PyTorch 2.5.1 (or higher) via `pip` following https://pytorch.org/. If you have a PyTorch version lower than 2.5.1 in your current environment, the installation command above will try to upgrade it to the latest PyTorch version using `pip`.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.
3. If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2 (some post-processing functionality may be limited, but it doesn't affect the results in most cases).



## Run the pipeline on a single scene

In this section, we provide information on how to run the pipeline for a single scene. In particular, we divide this section into four parts:
1. Download **checkpoints**
2. Check the format of **scene's data**
3. Set up **configurations** 
4. **Run** FlyMeThrough 

### Step 1: Download Checkpoints

First, we need to download the model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd sam2/checkpoints && \
./download_ckpts.sh && \
cd ../..

cd ml-depth-pro
source get_pretrained_models.sh  # Files will be downloaded to `checkpoints` directory.
cd ..
```

or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

### Step 2: Check the folder structure of the data for your scene 
In order to run FlyMeThrough you need to have access to the point cloud of the scene as well to the posed RGB-D frames.

We recommend creating a folder `scene_example` inside the `flymethrough` folder where the data is saved. 
```
scene_example
      ├── pose                            <- folder with camera poses
      │      ├── 0.txt 
      │      ├── 1.txt 
      │      └── ...  
      ├── frame                           <- folder with RGB images
      │      ├── 0.jpg (or .png/.jpeg)
      │      ├── 1.jpg (or .png/.jpeg)
      │      └── ...  
      ├── depth                           <- folder with depth images
      │      ├── 0.png (or .jpg/.jpeg)
      │      ├── 1.png (or .jpg/.jpeg)
      │      └── ...  
      ├── intrinsic                 
      │      └── intrinsic_color.txt       <- camera intrinsics
      └── scene_example.ply                <- point cloud of the scene
```

Please note the followings:
* The **point cloud** should be provided as a `.ply` file and the points are expected to be in the z-up right-handed coordinate system.
* The **camera intrinsics** and **camera poses** should be provided in a `.txt` file, containing a 4x4 matrix.
* The **RGB images** and the **depths** can be either in `.png`, `.jpg`, `.jpeg` format; the used format should be specified as explained in **Step 3**.
* The **RGB images** and their corresponding **depths** and **camera poses** should be named as `{FRAME_ID}.extension`, without zero padding for the frame ID, starting from index 0.


## Acknowledgments

We would like to thank the authors of [SAM2](https://github.com/facebookresearch/sam2) and [Depth-Pro](https://github.com/apple/ml-depth-pro) for their works which were used for our model.



## Citing FlyMeThrough

If you use FlyMeThrough or the structure in your research, please use the following BibTeX entry.

```bibtex
@inproceedings{su2025flymethrough,
  author       = {Xia Su and others},
  title        = {FlyMeThrough: Human-AI Collaborative 3D Indoor Mapping with Commodity Drones},
  booktitle    = {Proceedings of the 38th Annual ACM Symposium on User Interface Software and Technology (UIST ’25)},
  year         = {2025},
  doi          = {10.1145/XXXXX},
  publisher    = {ACM},
  location     = {Busan, Republic of Korea}
}
```

## TODO

We plan to release the complete scripts to this repository shortly, including:  
- The annotation and visualization interfaces code of *FlyMeThrough*.  
- The algorithms and implementation for *FlyMeThrough* 3D mapping of indoor spaces.

