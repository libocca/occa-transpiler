# HIP setup
This instruction explains how to install HIP, and make it work with nvidia runtime
Original AMD instructions are [here](https://github.com/ROCm/HIP/blob/develop/docs/how_to_guides/install.md)

## Requirements 
- Ubuntu 22.04

## Installation
**If you want to use Nvidia runtime:**
- Make sure you have nvidia driver and CUDA installed
  - Follow instructions on CUDA installation. https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ 
  - HIP expects cuda to be installed under `/usr/local/cuda`

```bash
sudo apt update && sudo apt upgrade
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms
sudo apt install rocm-hip-libraries
# reboot
sudo apt install hip-runtime-nvidia     # This should add possibility to compile for Nvidia
sudo apt install hip-dev
```

- According to [official instructions](https://github.com/ROCm/HIP/blob/develop/docs/how_to_guides/install.md), HIP is installed to (`/opt/rocm/hip`), but in reallity it is under `/opt/rocm/`
- All ROCm and HIP binaries are under `/opt/rocm/bin`
- You can remove `hip-runtime-amd` package if you are planning to use nvidia runtime only (this will save you ~14Gb)
**Important**
- Sometimes HIP can default to amd backend (even if you remove `hip-runtime-amd`). To enfornce nvidia backend use this ENV variable:
```bash
export HIP_PLATFORM=nvidia
```

## Verify installation
- Verify hip installation: `/opt/rocm/bin/hipconfig  --all`
- Verify that hip can compile and execute code:
```bash
git clone https://github.com/ROCm/HIP-Examples
cd HIP-Examples
git submodule update --init
cd vectorAdd && make
```
