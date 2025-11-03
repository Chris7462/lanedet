# LaneDet
## Introduction
LaneDet is an open source lane detection toolbox based on PyTorch that aims to pull together a wide variety of state-of-the-art lane detection models. Developers can reproduce these SOTA methods and build their own methods.

![demo image](.github/_clips_0601_1494452613491980502_20.jpg)

## Table of Contents
* [Introduction](#Introduction)
* [Benchmark and model zoo](#Benchmark-and-model-zoo)
* [Installation](#Installation)
* [Getting Started](#Getting-started)
* [Contributing](#Contributing)
* [Licenses](#Licenses)
* [Acknowledgement](#Acknowledgement)

## Benchmark and model zoo
Supported backbones:
- [x] ResNet
- [x] ERFNet
- [x] VGG
- [x] MobileNet
- [ ] DLA (coming soon)

Supported detectors:
- [x] [SCNN](configs/scnn)
- [x] [UFLD](configs/ufld)
- [x] [RESA](configs/resa)
- [x] [LaneATT](configs/laneatt)
- [x] [CondLane](configs/condlane)
- [ ] CLRNet (coming soon)


## Installation
<!--
Please refer to [INSTALL.md](INSTALL.md) for installation.
-->

### Clone this repository
```bash
git clone https://github.com/turoad/lanedet.git
cd lanedet
```
We call this directory as `$LANEDET_ROOT`

### Create a conda virtual environment and activate it (conda is optional)

```bash
conda create -n lanedet python=3.12 -y
conda activate lanedet
```

### Install dependencies

#### PyTorch 2.9.0 with CUDA 13.0 (Recommended)

```bash
# Install PyTorch 2.9.0 with CUDA 13.0 support
pip install torch==2.9.0+cu130 torchvision==0.24.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# Or if using conda:
# conda install pytorch==2.9.0 torchvision==0.24.0 pytorch-cuda=13.0 -c pytorch -c nvidia

# Install other Python packages
pip install -r requirements.txt

# Build and install LaneDet (this will compile CUDA extensions)
python setup.py build develop
```

#### Legacy PyTorch 1.8.0 (Not recommended)

> **Note**: This version uses an older PyTorch version and is no longer actively maintained. 
> For new installations, please use PyTorch 2.9.0 as shown above.

<details>
<summary>Click to expand legacy installation instructions</summary>

```bash
# Install pytorch firstly, the cudatoolkit version should be same in your system.
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.1 -c pytorch

# Or you can install via pip
pip install torch==1.8.0 torchvision==0.9.0

# Install python packages
python setup.py build develop
```
</details>

### Verify Installation

```bash
# Test if CUDA extensions are properly built
python -c "from lanedet.ops import nms; print('✓ NMS import successful!')"

# Check PyTorch and CUDA versions
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### System Requirements

- **Python**: 3.10 or higher (3.8+ for PyTorch 1.8.0)
- **PyTorch**: 2.9.0+cu130 (or 1.8.0 for legacy)
- **CUDA**: 13.0 (or 10.1 for legacy PyTorch 1.8.0)
- **GPU**: NVIDIA GPU with CUDA support
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Troubleshooting

#### CUDA Extension Build Failures

If you encounter errors during `python setup.py build develop`:

1. Ensure CUDA toolkit is properly installed and matches your PyTorch version
2. Check that `nvcc` is in your PATH: `nvcc --version`
3. Verify your GPU driver supports your CUDA version
4. Try cleaning build artifacts: `rm -rf build/ lanedet.egg-info/` then rebuild

#### Import Errors

If you see import errors related to `nms_impl`:
```bash
# Rebuild the extensions
python setup.py build develop --force
```

#### Runtime CUDA Errors

If you encounter CUDA errors during training/inference:
1. Check GPU memory: `nvidia-smi`
2. Verify CUDA version matches: `python -c "import torch; print(torch.version.cuda)"`
3. Try with a smaller batch size first

### Data preparation

#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```bash
cd $LANEDET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```

#### TuSimple
Download [TuSimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```bash
cd $LANEDET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For TuSimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file
```

For TuSimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```bash
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# This will generate seg_label directory
```

## Getting Started

### Training

For training, run

```bash
python main.py [configs/path_to_your_config] --gpus [gpu_ids]
```

For example, run
```bash
python main.py configs/resa/resa50_culane.py --gpus 0
```

### Testing

For testing, run
```bash
python main.py [configs/path_to_your_config] --validate --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, run
```bash
python main.py configs/resa/resa50_culane.py --validate --load_from culane_resnet50.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.

For example, run
```bash
python main.py configs/resa/resa50_culane.py --validate --load_from culane_resnet50.pth --gpus 0 --view
```

### Inference

See `tools/detect.py` for detailed information.
```bash
python tools/detect.py --help

usage: detect.py [-h] [--img IMG] [--show] [--savedir SAVEDIR]
                 [--load_from LOAD_FROM]
                 config

positional arguments:
  config                The path of config file

optional arguments:
  -h, --help            show this help message and exit
  --img IMG             The path of the img (img file or img_folder), for
                        example: data/*.png
  --show                Whether to show the image
  --savedir SAVEDIR     The root of save directory
  --load_from LOAD_FROM
                        The path of model
```

To run inference on example images in `./images` and save the visualization images in `vis` folder:
```bash
python tools/detect.py configs/resa/resa34_culane.py --img images/ \
      --load_from resa_r34_culane.pth --savedir ./vis
```

## Contributing

We appreciate all contributions to improve LaneDet. Any pull requests or issues are welcomed.

## Licenses

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

<!--ts-->
* [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
* [pytorch/vision](https://github.com/pytorch/vision)
* [cardwing/Codes-for-Lane-Detection](https://github.com/cardwing/Codes-for-Lane-Detection)
* [XingangPan/SCNN](https://github.com/XingangPan/SCNN)
* [ZJULearning/resa](https://github.com/ZJULearning/resa)
* [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
* [lucastabelini/LaneATT](https://github.com/lucastabelini/LaneATT)
* [aliyun/conditional-lane-detection](https://github.com/aliyun/conditional-lane-detection)
<!--te-->

## Migration Notes

### PyTorch 2.9.0 Migration (November 2025)

This repository has been updated to support PyTorch 2.9.0 with CUDA 13.0. Key changes include:

- Updated CUDA extension APIs to PyTorch 2.x standards
- Replaced deprecated functions (`F.upsample` → `F.interpolate`)
- Modernized C++ extension code (removed deprecated macros)

**Important**: This version is **NOT backward compatible** with PyTorch 1.8.0. If you need to use PyTorch 1.8.0, please checkout an earlier commit or use the legacy installation instructions above.

For detailed migration information, see the migration documentation in the repository.

<!-- 
## Citation

If you use LaneDet in your research, please cite:

```bibtex
@misc{zheng2021lanedet,
  author =       {Tu Zheng},
  title =        {LaneDet},
  howpublished = {\url{https://github.com/turoad/lanedet}},
  year =         {2021}
}
```
-->
