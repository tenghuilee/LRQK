import argparse
import copy
import fnmatch
import math
import os
import os.path as osp
import statistics
import sys
import time
from collections import Counter
from inspect import signature
from typing import List

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from opencompass.registry import (ICL_EVALUATORS, MODELS, TASKS,
                                  TEXT_POSTPROCESSORS)
from opencompass.tasks.openicl_eval import OpenICLEvalTask
from opencompass.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_output_path, get_logger,
                               task_abbr_from_cfg)


@TASKS.register_module()
class WrapedOpenICLEvalTask(OpenICLEvalTask):
    """OpenICL Evaluation Task.

    This task is used to evaluate the metric between predictions and
    references.
    """

    def get_command(self, cfg_path, template):
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        command = f'{python} {script_path} {cfg_path}'
        return template.format(task_cmd=command)

def parse_args():
    parser = argparse.ArgumentParser(description='Score Calculator')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.time()
    inferencer = WrapedOpenICLEvalTask(cfg)
    inferencer.run()
    end_time = time.time()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
