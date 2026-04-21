# FlashFPS-Openpoints

FlashFPS for the PointNeXt / PointVector models, built on the [openpoints](https://github.com/guochengqian/openpoints) backbone. This sub-README covers environment setup, pretrained weights, dataset preparation, and evaluation commands for the **S3DIS** and **ScanNet** segmentation benchmarks.

## 1. Installation

First clone the repository:

```bash
git clone https://github.com/Yuzhe-Fu/FlashFPS
cd FlashFPS
```

### 1.1 Environment Setup

We reuse the environment from [HPCA'26-FractalCloud](https://github.com/Yuzhe-Fu/FractalCloud). Two options are supported: **Docker (recommended)** or local installation.

#### Option 1: Docker (recommended)

> Time: 20–30 min download + 5–10 min one-click setup.

Download the image (≈45 GB). HuggingFace is more stable; Google Drive is a mirror.

```bash
# Option A — HuggingFace (recommended)
wget https://huggingface.co/YuzheFu/FractalCloud/resolve/main/FractalCloud_docker.tar

# Option B — Google Drive mirror
gdown --fuzzy "https://drive.google.com/file/d/1bjkS6beJeIV8MLgCd0CKbMack_s5fmAt/view"
```

Import the Docker image (make sure Docker is already installed on the host):

```bash
docker import FractalCloud_docker.tar flashfps_env:base
```

Start the container from the **FlashFPS repo root**:

```bash
# Run this from the root of the cloned FlashFPS/ repo.
docker run --name flashfps \
  -it --gpus all --shm-size 32G \
  -v $(pwd):/workspace \
  flashfps_env:base \
  /bin/bash
```

You may see a `command not found` message when the shell starts — this can be safely ignored. The container automatically activates the `openpoints` conda environment with all dependencies installed.

If you want to support the SOTA work **QuickFPS**, please build its CUDA extension by:

```bash
cd FlashFPS-Openpoints/openpoints/cpp/pointnet2_batch
python setup.py install
cd ../../..
```

> Note: After the first `docker run`, you can re-enter the same Docker container with:
>
> ```bash
> docker exec -it flashfps /bin/bash
> ```

#### Option 2: Local installation

> Time: 30 min – 1.5 h, depending on your server environment.
> **Root permission is required** to install the PointNeXt library.

We recommend CUDA 11.x (tested with CUDA 11.3), as required by [OpenPoints](https://github.com/guochengqian/PointNeXt). Other CUDA major versions may cause installation or runtime failures. Check your CUDA version with `nvcc --version`.

To set up a compatible CUDA 11.3 toolchain, we recommend using Anaconda:

```bash
conda install -y cuda=11.3.1 -c nvidia/label/cuda-11.3.1
```

Our install script targets CUDA 11.3. If you use a different CUDA 11.x version, adjust the script accordingly. The script will:

- Check whether `conda` is available.
- Create a dedicated conda environment (`openpoints`).
- Install PyTorch and a matching CUDA runtime automatically.

```bash
source install.sh
```

> If you run into installation issues, the [PointNeXt troubleshooting thread](https://github.com/guochengqian/PointNeXt/issues) is a good first stop.


### 1.2 Pretrained Models

All commands in the following sections assume you are at the:
- `./workspace/FlashFPS-Openpoints` (Docker setup), or
- `./FlashFPS/FlashFPS-Openpoints` (local installation)


To download the **pretrained weights**, make sure `gdown` is available (it is already installed in our Docker image and via `install.sh`). Otherwise install it manually: `pip install gdown`

Then download the pretrained models:

```bash
mkdir -p ./log
cd ./log
gdown --folder https://drive.google.com/drive/folders/1ChRWL8lk5bidxpub1OXAX1sAdZlFlFj-
cd ..
```

Alternatively, you can download manually from [Google Drive](https://drive.google.com/drive/folders/1ChRWL8lk5bidxpub1OXAX1sAdZlFlFj-?usp=sharing). Please place files into the `./log` folder. The folder structure should be like:

```
./log
├── Reported_Logs
│   ├── s3dis
│   │   ├── pointnext-l
|   |   │   │   ├── ...
│   │   ├── pointvector-l
|   |   │   │   ├── ...
│   ├── scannet
│   │   ├── pointnext-l
|   |   │   │   ├── ...
│   │   ├── pointvector-l
|   |   │   │   ├── ...
```

We also provide the **evaluation logs** (e.g., '0-Reported-FlashFPS-xx%.log') for all evaluated checkpoints. They correspond to the results reported in the **Table 2** of the paper and can be used to sanity-check your own reproduction results.

### 1.3 Dataset Preparation

```bash
source ./script/download_DS.sh
```

## 2. Experiments

Please run the following commands in the:

- `./workspace/FlashFPS-Openpoints` (Docker setup), or
- `./FlashFPS/FlashFPS-Openpoints` (local installation)

For configuration details: 

- The `flashfps.useFlashFPS` flag controls whether to use FlashFPS. Set it to `True` to use FlashFPS.
- The `flashfps.PruneRate` flag controls the prune ratio. Set it to **0, 0.25, 0.5, or 0.75** to reproduce the corresponding results of **Table 2** in the paper.

> **Tip.** Do not copy and paste the following commands into a single line.

### 2.1 S3DIS Segmentation (PointNeXt-L)

```bash
# Baseline — FPS-CUDA
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l.yaml \
  wandb.use_wandb=False mode=test \
  --pretrained_path ./log/Reported_Logs/s3dis/pointnext-l/checkpoint/s3dis-train-pointnext-l-ngpus1-seed6266-20220525-162629-D7sCFuHmsMP9Kk5bdAA2Td_ckpt_best.pth

# FlashFPS
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l.yaml \
  wandb.use_wandb=False mode=test \
  flashfps.useFlashFPS=True \
  flashfps.useFPS_Prune=True \
  flashfps.useFPS_Cache=True \
  flashfps.PruneRate=0.75 \
  --pretrained_path ./log/Reported_Logs/s3dis/pointnext-l/checkpoint/s3dis-train-pointnext-l-ngpus1-seed6266-20220525-162629-D7sCFuHmsMP9Kk5bdAA2Td_ckpt_best.pth
```

### 2.2 S3DIS Segmentation (PointVector-L)

```bash
# Baseline — FPS-CUDA
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointvector-l.yaml \
  wandb.use_wandb=False mode=test \
  --pretrained_path ./log/Reported_Logs/s3dis/pointvector-l/checkpoint/s3dis-train-pointvector-l-ngpus1-20250303-095423-MgnzawcdpjwKCqpsRoovPf_ckpt_best.pth

# FlashFPS
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointvector-l.yaml \
  wandb.use_wandb=False mode=test \
  flashfps.useFlashFPS=True \
  flashfps.useFPS_Prune=True \
  flashfps.useFPS_Cache=True \
  flashfps.PruneRate=0.75 \
  --pretrained_path ./log/Reported_Logs/s3dis/pointvector-l/checkpoint/s3dis-train-pointvector-l-ngpus1-20250303-095423-MgnzawcdpjwKCqpsRoovPf_ckpt_best.pth
```

### 2.3 ScanNet Segmentation (PointNeXt-L)

```bash
# Baseline — FPS-CUDA
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/scannet/pointnext-l.yaml \
  wandb.use_wandb=False mode=test \
  --pretrained_path ./log/Reported_Logs/scannet/pointnext-l/checkpoint/scannet-train-pointnext-l-ngpus2-20250703-134408-YjMWZ22Hm6ghZEokew2yYn_ckpt_best.pth

# FlashFPS
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/scannet/pointnext-l.yaml \
  wandb.use_wandb=False mode=test \
  flashfps.useFlashFPS=True \
  flashfps.useFPS_Prune=True \
  flashfps.useFPS_Cache=True \
  flashfps.PruneRate=0.75 \
  --pretrained_path ./log/Reported_Logs/scannet/pointnext-l/checkpoint/scannet-train-pointnext-l-ngpus2-20250703-134408-YjMWZ22Hm6ghZEokew2yYn_ckpt_best.pth
```

### 2.4 ScanNet Segmentation (PointVector-L)

```bash
# Baseline — FPS-CUDA
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/scannet/pointvector-l.yaml \
  wandb.use_wandb=False mode=test \
  --pretrained_path ./log/Reported_Logs/scannet/pointvector-l/checkpoint/scannet-train-pointvector-l-ngpus2-20250704-125023-LU4wubxEbCFG2nd7uyfbjf_ckpt_best.pth

# FlashFPS
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/scannet/pointvector-l.yaml \
  wandb.use_wandb=False mode=test \
  flashfps.useFlashFPS=True \
  flashfps.useFPS_Prune=True \
  flashfps.useFPS_Cache=True \
  flashfps.PruneRate=0.75 \
  --pretrained_path ./log/Reported_Logs/scannet/pointvector-l/checkpoint/scannet-train-pointvector-l-ngpus2-20250704-125023-LU4wubxEbCFG2nd7uyfbjf_ckpt_best.pth
```

### 2.5 SOTA baseline: QuickFPS

To reproduce the QuickFPS performance in **Fig. 7** of the paper, simply set `flashfps.enable_quick_fps=True` in the command (leave the other `flashfps.`* flags at their defaults, i.e., FlashFPS off).

For example, QuickFPS with PointNeXt-L on S3DIS:

```bash
CUDA_VISIBLE_DEVICES=0 bash script/main_segmentation.sh cfgs/s3dis/pointnext-l.yaml \
  wandb.use_wandb=False mode=test \
  flashfps.enable_quick_fps=True \
  --pretrained_path ./log/Reported_Logs/s3dis/pointnext-l/checkpoint/s3dis-train-pointnext-l-ngpus1-seed6266-20220525-162629-D7sCFuHmsMP9Kk5bdAA2Td_ckpt_best.pth
```

## Some Notes:

1. Some frequent commands for docker usage

```bash
exit                               # exit Docker container
docker start flashfps          # start container
docker exec -it flashfps /bin/bash   # attach interactive shell
docker stop flashfps           # stop container
```

