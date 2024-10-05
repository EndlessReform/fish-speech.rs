## Building from source in Windows (recommended)

Open PowerShell.

1. Install latest Nvidia drivers with [GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/).
2. If you don't have Rust, follow the official [Rust Windows install steps](https://www.rust-lang.org/tools/install). Open PowerShell and test that it works, with command:

```
rustc --version
```

3. Install CUDA with the [Nvidia Toolkit for Windows](https://developer.nvidia.com/cuda-downloads). You can test that it works in PowerShell with:

```
# Should return something like 12.6
nvcc --version
```

4. Install [Git for Windows](https://gitforwindows.org) if you haven't already.
5. Clone the repo to wherever you want:

```
# In the directory you want
git clone git@github.com:EndlessReform/fish-speech.rs.git
```

7. Download the weights:

```
mkdir checkpoints
cd checkpoints
git lfs install
git clone https://huggingface.co/jkeisling/fish-speech-1.4
```

Now from the Start menu, search for the program "x64 Native Tools Command Prompt for VS 2022". Open it and run all the below commands from there. It is VERY important that you do this; if you don't, the program will NOT compile.

8. Build the program as normal:

```
cargo build --release --features cuda
```

9. TODO

## Direct install in WSL (not recommended)

> [!WARNING]
> WSL is a LOT slower than native Linux: https://github.com/huggingface/candle/issues/1829. With Fish 1.4, I've verified it to be up to 3x slower on the same hardware. Only do this if you REALLY want to.

This guide assumes you have WSL already installed, with hardware
