import os
import os.path as osp
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from typing import Any, Dict, List, Tuple

import mmengine
import numpy as np
from mmengine.config import ConfigDict
from mmengine.device import is_npu_available
from tqdm import tqdm

from opencompass.registry import RUNNERS, TASKS
from opencompass.utils import get_logger, model_abbr_from_cfg

from opencompass.runners.local import get_command_template as official_get_command_template
from opencompass.runners.local import LocalRunner


def get_command_template(gpu_ids: List[int]) -> str:
    _path = osp.join(osp.dirname(__file__), '..')
    _path = osp.abspath(_path)
    _cmd = official_get_command_template(gpu_ids)
    
    _cmd = 'export PYTHONPATH="'+ _path + '":$PYTHONPATH; ' + _cmd
    return _cmd

@RUNNERS.register_module()
class WrapedLocalRunner(LocalRunner):
    """Local runner. Start tasks by local python.

    Args:
        task (ConfigDict): Task type config.
        max_num_workers (int): Max number of workers to run in parallel.
            Defaults to 16.
        max_workers_per_gpu (int): Max number of workers to run for one GPU.
            Defaults to 1.
        debug (bool): Whether to run in debug mode.
        lark_bot_url (str): Lark bot url.
    """

    def _launch(self, task, gpu_ids, index):
        """Launch a single task.

        Args:
            task (BaseTask): Task to launch.

        Returns:
            tuple[str, int]: Task name and exit code.
        """

        task_name = task.name

        pwd = os.getcwd()
        # Dump task config to file
        mmengine.mkdir_or_exist('tmp/')
        # Using uuid to avoid filename conflict
        import uuid
        uuid_str = str(uuid.uuid4())
        param_file = f'{pwd}/tmp/{uuid_str}_params.py'

        logger = get_logger()
        try:
            task.cfg.dump(param_file)
            tmpl = get_command_template(gpu_ids)
            get_cmd = partial(task.get_command,
                              cfg_path=param_file,
                              template=tmpl)
            cmd = get_cmd()

            logger.debug(f'Running command: {cmd}')

            # Run command
            out_path = task.get_log_path(file_extension='out')
            mmengine.mkdir_or_exist(osp.split(out_path)[0])
            stdout = open(out_path, 'w', encoding='utf-8')

            result = subprocess.run(cmd,
                                    shell=True,
                                    text=True,
                                    stdout=stdout,
                                    stderr=stdout)

            if result.returncode != 0:
                logger.error(f'task {task_name} fail, see\n{out_path}')
                with open(out_path, 'r', encoding='utf-8') as f:
                    logger.error(f'output: {f.read()}')
        finally:
            # Clean up
            if not self.keep_tmp_file:
                os.remove(param_file)
            else:
                pass
        return task_name, result.returncode
