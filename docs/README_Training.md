# JarvisArt Training Guide

JarvisArt training consists of two stages: **SFT** and **GRPO-R**.


## SFT Training

Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training. Refer to [Qwen2.5-VL Examples](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_full/qwen2_5vl_full_sft.yaml).

See [data_scripts](../data_scripts/README.md) for data generation pipeline.

## GRPO-R Training
### System Architecture

<div align="center">

GRPO-R training uses a distributed architecture with the following components:

```
┌─────────────────────────────────────┐
│         Linux Training Server       │
│  ┌─────────────────────────────┐    │
│  │   GRPO-R Training Script    │    │
│  │ (src/grpo-r/run_scripts)    │    │
│  └──────────────┬──────────────┘    │
│                 │                   │
│  ┌──────────────▼──────────────┐    │
│  │      LRC Task Server        │    │
│  │   (lrc_scripts/servers)     │    │
│  └──────────────┬──────────────┘    │
└─────────────────┼───────────────────┘
│                 │    HTTP API       │
┌─────────────────▼───────────────────┐
│         Local Client Machine(s)     │
│  ┌─────────────────────────────┐    │
│  │      LRC Task Client        │    │
│  │   (lrc_scripts/clients)     │    │
│  └──────────────┬──────────────┘    │
│                 │                   │
│  ┌──────────────▼──────────────┐    │
│  │ Adobe Lightroom Classic     │    │
│  │  + XMPlayer Plugin          │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

</div>

### 1. Environment Setup

**Linux Server:**
```bash
conda env create -f ../envs/environment_rl.yml
conda activate jarvisart_rl
```

**Mac Client:**
```bash
pip install aiohttp fastapi uvicorn requests pillow pyyaml
```

**Install Lightroom Plugin:**
Open Lightroom Classic → `File` → `Plug-in Manager` → `Add` → Select `lrc_scripts/clients/agent_to_lightroom/XMPlayer.lrplugin/`

### 2. Start Services (Execute in Order)

**Step 1: Start LRC Server on Linux Server**
```bash
cd lrc_scripts/servers
./start_reverse_server.sh
```

**Step 2: Start LRC Client on Mac Client**
```bash
cd lrc_scripts/clients
./start_mac_client.sh --servers "LINUX_SERVER_IP:8081"
```
> Ensure Lightroom Classic is running and XMPlayer plugin is installed

**Step 3: Start Training on Linux Server**
```bash
cd src/grpo-r/run_scripts
./run_grpo.sh
```

### 3. Configuration Files

**`run_grpo.sh`** - Update the following paths:
```bash
WORK_DIR="JarvisArt/src/grpo-r"
SAVE_CKPT_PATH="JarvisArt/checkpoints/jarvisart_rl"
SCRIPT_PATH="JarvisArt/src/grpo-r/run_scripts/training_args.yaml"
```

**`training_args.yaml`** - Update the following paths:
```yaml
model_name_or_path: "/path/to/JarvisArt/"
data_file_paths: "/path/to/data/grpo_dataset.json"
lightroom_upload_dir: "/path/to/lightroom/uploads"
lightroom_results_dir: "/path/to/lightroom/results"
lightroom_temp_dir: "/path/to/lightroom/lightroom_temp"
```
