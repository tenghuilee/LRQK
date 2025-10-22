# ShadowKV

original: [https://github.com/ByteDance-Seed/ShadowKV.git](https://github.com/ByteDance-Seed/ShadowKV.git)

## build

```shell
# flashinfer; choose base on your cuda versoin and pytorch version
# goto: https://flashinfer.ai/whl/ for more verions
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5

# cutlass
mkdir shadowkv_lib/3rdparty
git clone https://github.com/NVIDIA/cutlass.git 3rdparty/cutlass

# build
python setup.py build

# ln the build kernel to shadowkv_lib
# in dir shadowkv_lib
ln -s build/lib.<path-to-build>/<path-to-so>.so .
```

## How to use?

