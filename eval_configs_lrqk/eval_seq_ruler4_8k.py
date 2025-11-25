# import os.path

from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate

with read_base():
    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_128k import needlebench_128k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k import needlebench_datasets as needlebench_8k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_datasets as needlebench_32k_datasets

    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_4k_datasets
    # from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_8k_datasets
    # from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_datasets
    # from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_32k_datasets

# datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

datasets = []

for ds in ruler_4k_datasets:
    abbr = ds["abbr"]
    if "qa" in abbr or "vt" in abbr:
        datasets.append(ds)

from opencompass.partitioners import NumWorkerPartitioner

models = [
    dict(
        type="LRQKChatBot",
        abbr=f"llama-3-8B-1M_LRQKChatBot_r16_act1024_lite64",
        path="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        lrqk_rank = 16,
        lrqk_num_active_tokens = 1024,
        lrqk_lite_tokens = 64,
    ),
    dict(
        type="LRQKChatBot",
        abbr=f"Qwen2.5-7B_LRQKChatBot_r16_act1024_lite64",
        path="Qwen/Qwen2.5-7B-Instruct",
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        lrqk_rank = 16,
        lrqk_num_active_tokens = 1024,
        lrqk_lite_tokens = 64,
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="llama-3-8B-1M",
        path="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        max_out_len=1024,
        batch_size=2,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-7b-instruct-hf',
        path='Qwen/Qwen2.5-7B-Instruct',
        max_out_len=4096,
        batch_size=2,
        run_cfg=dict(num_gpus=1),
    )
]

# Local Runner
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        # num_worker=1,
    ),
    runner=dict(
        type="WrapedLocalRunner",
        max_num_workers=32,
        retry=5,  # Modify if needed
        task=dict(type="WrapedOpenICLInferTask")
    ),
)

