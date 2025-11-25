# import os.path

from mmengine.config import read_base

with read_base():

    # from opencompass.configs.datasets.needlebench.needlebench_128k import needlebench_128k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k import needlebench_datasets as needlebench_8k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_datasets as needlebench_32k_datasets

    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

    # from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_4k_datasets
    # from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_8k_datasets
    # from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_datasets
    # from opencompass.configs.datasets.ruler.ruler_128k_gen import ruler_datasets as ruler_128k_datasets
    from opencompass.configs.datasets.longbench.longbench import longbench_datasets

pre_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
datasets = []

_picked_list = [
    "narrativeqa",
    "multifieldqa",
    "gov_report",
    "samsum",
    "passage_retrieval",
    "lcc",
]

for ds in pre_datasets:
    abbr = ds['abbr'].lower()
    for _pl in _picked_list:
        if _pl in abbr:
            datasets.append(ds)
            break

del _picked_list
del pre_datasets

from opencompass.partitioners import NumWorkerPartitioner

models = [
    # dict(
    #     type="LRQKChatBot",
    #     abbr=f"llama-3-8B-1M_LRQKChatBot_r16_act2048_lite64",
    #     path="gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    #     max_out_len=1024,
    #     batch_size=1,
    #     run_cfg=dict(num_gpus=1),
    #     lrqk_rank = 16,
    #     lrqk_num_active_tokens = 2048,
    #     lrqk_lite_tokens = 64,
    # ),
    dict(
        type="LRQKChatBot",
        abbr=f"llama-3.1-8B-1M_LRQKChatBot_r16_act1024_lite64",
        path="./llama_hf_mirror/llama_hf/Meta-Llama-3.1-8B-Instruct",
        max_out_len=512,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        lrqk_rank = 16,
        lrqk_num_active_tokens = 1024,
        lrqk_lite_tokens = 64,
    ),
]

# Local Runner
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        # num_worker=8,
    ),
    runner=dict(
        type="WrapedLocalRunner",
        max_num_workers=32,
        retry=5,  # Modify if needed
        task=dict(type="WrapedOpenICLInferTask")
    ),
)

