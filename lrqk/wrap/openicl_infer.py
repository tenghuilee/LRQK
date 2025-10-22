import argparse
import os
import os.path as osp
import random
import sys
import time
from typing import Any

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_INFERENCERS, ICL_PROMPT_TEMPLATES,
                                  ICL_RETRIEVERS, TASKS)
from opencompass.tasks.openicl_infer import OpenICLInferTask
from opencompass.utils import (build_dataset_from_cfg, build_model_from_cfg,
                               get_infer_output_path, get_logger,
                               model_abbr_from_cfg, task_abbr_from_cfg)

import lrqk.wrap.chat_kvquant
import lrqk.wrap.chat_lrqk
import lrqk.wrap.chat_empty
import lrqk.wrap.subset_gsm8k

@TASKS.register_module()
class WrapedOpenICLInferTask(OpenICLInferTask):

    def get_command(self, cfg_path, template):
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template which have '{task_cmd}' to format
                the command.
        """
        sys.path.append(os.getcwd())
        script_path = __file__
        backend_keys = ['VLLM', 'Lmdeploy']
        use_backend = any(
            key in str(self.model_cfgs[0].get('type', ''))
            or key in str(self.model_cfgs[0].get('llm', {}).get('type', ''))
            for key in backend_keys)
        if self.num_gpus > 1 and not use_backend:
            port = random.randint(12000, 32000)
            command = (f'torchrun --master_port={port} '
                       f'--nproc_per_node {self.num_procs} '
                       f'{script_path} {cfg_path}')
        else:
            python = sys.executable
            command = f'{python} {script_path} {cfg_path}'

        return template.format(task_cmd=command)

def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = WrapedOpenICLInferTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
