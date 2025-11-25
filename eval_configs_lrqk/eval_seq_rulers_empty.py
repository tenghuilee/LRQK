# import os.path

from mmengine.config import read_base

with read_base():

    # from opencompass.configs.datasets.needlebench.needlebench_128k import needlebench_128k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_8k.needlebench_8k import needlebench_datasets as needlebench_8k_datasets

    # from opencompass.configs.datasets.needlebench.needlebench_32k.needlebench_32k import needlebench_datasets as needlebench_32k_datasets

    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_4k_datasets
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_8k_datasets
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_datasets
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_32k_datasets
    from opencompass.configs.datasets.ruler.ruler_64k_gen import ruler_datasets as ruler_64k_datasets
    from opencompass.configs.datasets.ruler.ruler_128k_gen import ruler_datasets as ruler_128k_datasets
    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

from opencompass.partitioners import NumWorkerPartitioner

models = [
    dict(
        type="EmptyChatBot",
        abbr=f"empty",
        path="./llama_hf/Meta-Llama-3.1-8B-Instruct",
        max_out_len=125,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]

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

