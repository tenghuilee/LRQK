
# 

import subprocess
import time
import torch
import multiprocessing as mp
import os


num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")

# Run the command to estimate the time per step

task_list = [
    "lrqk_default",
    "lrqk_offload",
    "lrqk_no_hitmiss",
    "gpu_only",
    "cpu_offload",
]

def estimate_time_per_step(args_tuple: tuple):
    device, task = args_tuple
    print(f"Estimating time per step for {task} on GPU {device}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    subprocess.run([
            "python",
            "tps_estimate.py",
            "--task", task,
        ],
        env=env,
    )
    # time.sleep(4)

cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

if cuda_devices is None:
   # Create a pool of workers
   num_gpus = torch.cuda.device_count()
   cuda_devices = [str(i) for i in range(num_gpus)]
else:
    cuda_devices = [s.strip() for s in cuda_devices.split(",")]
    num_gpus = len(cuda_devices)

with mp.Pool(num_gpus) as pool:
    results = pool.map(
        estimate_time_per_step,
        [(cuda_devices[i%num_gpus], task) for i, task in enumerate(task_list)],
    )

print("All tasks completed")
