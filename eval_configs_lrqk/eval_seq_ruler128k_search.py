# import os.path

from opencompass.partitioners import NumWorkerPartitioner
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.longbench.longbench import longbench_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_128k import needlebench_128k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k import needlebench_datasets as needlebench_8k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_datasets as needlebench_32k_datasets

    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

    # from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_4k_datasets
    # from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_8k_datasets
    # from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_datasets
    from opencompass.configs.datasets.ruler.ruler_128k_gen import ruler_datasets as ruler_128k_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

path_abbr_list = [
    (
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        "llama-3-8B-1M",
    ),
    (
        "THUDM/glm-4-9b-chat-1m",
        "glm-4-9b-chat-1m",
    ),
    (
        "/shared/home/xuyang/8_KVN/llama-3.1-8B/hf",
        "Llama-3.1-8B-Instruct",
    ),
]


models = []

for _r in [16, 32]:
    for _act in [4096, 2048]:
        for _p, _a in path_abbr_list:
            models.append(dict(
                type="LRQKChatBot",
                abbr=f"{_a}_LRQKChatBot_r{_r}_act{_act}_lite64",
                path=_p,
                max_out_len=1024,
                batch_size=1,
                run_cfg=dict(num_gpus=1),
                lrqk_rank=_r,
                lrqk_num_active_tokens=_act,
                lrqk_lite_tokens=64,
            ))


# Local Runner
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
    ),
    runner=dict(
        type="WrapedLocalRunner",
        max_num_workers=32,
        retry=5,  # Modify if needed
        task=dict(type="WrapedOpenICLInferTask")
    ),
)
