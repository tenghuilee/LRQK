# import os.path

from opencompass.partitioners import NumWorkerPartitioner
from mmengine.config import read_base
from opencompass.models import HuggingFacewithChatTemplate

with read_base():
    # from opencompass.configs.models.qwen2_5.hf_qwen2_5_7b_instruct
    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_128k import needlebench_128k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k import needlebench_datasets as needlebench_8k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_datasets as needlebench_32k_datasets

    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_4k_datasets
    # from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_8k_datasets
    # from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_datasets
    # from opencompass.configs.datasets.ruler.ruler_128k_gen import ruler_datasets as ruler_128k_datasets

# datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
datasets = []

for ds in ruler_4k_datasets:
    abbr = ds["abbr"]
    if "qa" in abbr or "vt" in abbr:
        datasets.append(ds)

path_abbr_list = [
    (
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "llama-3-8B-1M",
    ),
]


models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr=path_abbr_list[0][1],
        path=path_abbr_list[0][0],
        max_out_len=1024,
        batch_size=2,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
    ),
    # dict(
    #     type=HuggingFacewithChatTemplate,
    #     abbr='qwen2.5-7b-instruct-hf',
    #     path='Qwen/Qwen2.5-7B-Instruct',
    #     max_out_len=4096,
    #     batch_size=2,
    #     run_cfg=dict(num_gpus=1),
    # )
]

params = [
    (8, 256, 16),
    (8, 512, 16),
    (8, 1024, 16),
    (16, 256, 16),
    (24, 256, 16),
    (32, 256, 16),
]

for _r, _act, _lite in params:
    for _p, _a in path_abbr_list:
        models.append(dict(
            type="LRQKChatBot",
            abbr=f"{_a}_LRQKChatBot_r{_r}_act{_act}_lite{_lite}",
            path=_p,
            max_out_len=1024,
            batch_size=1,
            run_cfg=dict(num_gpus=1),
            lrqk_rank=_r,
            lrqk_num_active_tokens=_act,
            lrqk_lite_tokens=_lite,
        ))


# Local Runner
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
    ),
    runner=dict(
        type="WrapedLocalRunner",
        max_num_workers=32,
        retry=5,  # Modify if needed
        task=dict(type="WrapedOpenICLInferTask")
    ),
)
