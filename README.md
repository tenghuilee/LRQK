# Efficient Low Rank Attention for Long-Context Inference in Large Language Models

code updating ...

## Requirements

We tested the code on the following environment:

- Python: 3.10.16
- pytorch: 2.5.1
- huggingface-hub: 0.24.7 
- transformers: 4.47.1
- opencompass: 0.3.9; install via source code. Please refer to [opencompass.git](https://github.com/open-compass/opencompass.git); commit id: 7f2aeeff26bf550563092e8368bf63d5526fae26
- wonderwords: 2.2.0; for `RULER` benchmark

There is a cpp plugin that need to be compiled.
```bash
cd cpp_kernel
make
```


## Introduction

The main code is `lrqk_attention.py`. 
We test with batchsize=1.

### Demos

For a quick demo, please refer to [quick_demo.py](./quick_demo.py).
For a detailed demo, please refer to [demo_lrqk.py](./demo_lrqk.py).

### Evals

To evaluate with opencompass, please run 
```bash
# in the root directory
# set visible device in need
export CUDA_VISIBLE_DEVICES=<list of avaliables GPUs>
# Run the evaluation
python ./opencompass_run.py --reuse "latest" ./eval_configs/<configuration file.py>
# or 
./opencompass_run.py --reuse "latest" ./eval_configs/<configuration file.py>
# If this file has the permission to be executed.
```
where 
- `<list of avaliables GPUs>`: the GPUs to use. If you want to use all GPUs, please ignore `export CUDA_VISIBLE_DEVICES=...`.
- reuse "latest": reuse the latest results
- configuration file.py: the evaluation configuration
- for more commands, please refer to [opencompass](https://github.com/open-compass/opencompass.git) or run `python ./opencompass_run.py --help`.

It will automatically download the model and the evaluation data.
The results will be saved in `./outputs`.

### Additional WebUI

An additional WebUI is provided for visualizing the attention changnes. 
Run with 
```bash
# set visible device in need
export CUDA_VISIBLE_DEVICES=<list of avaliables GPUs>
# Run the webui
uvicorn webui_attention_track:app --host 0.0.0.0 --port 8089
```
You can access the webui at `http://localhost:8089/`.

Note: package `uvicorn` is required. Please install it via `pip install uvicorn`.


