
import argparse
import json
import os
import time
import threading
import subprocess
from typing import Optional, Tuple, Union, Dict, List, Any

import torch
import transformers
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

import shadowkv_lib
import lrqk_attention

def get_nvidia_smi_info():
    """Get GPU information using nvidia-smi command."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total,utilization.memory",
        "--format=csv,nounits,noheader"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"Error executing nvidia-smi: {result.stderr}")

    lines = result.stdout.strip().split('\n')
    nvidia_data = []
    for line in lines:
        fields = line.split(', ')
        if len(fields) != 8:
            continue
        gpu_info = {
            'index': int(fields[0]),
            'name': fields[1],
            'temperature.gpu': float(fields[2]),
            'power.draw': float(fields[3]),
            'utilization.gpu': int(fields[4].strip('%')),
            'memory.used': int(fields[5]) / 1024,  # MiB to GB
            'memory.total': int(fields[6]) / 1024,  # MiB to GB
            'utilization.memory': int(fields[7].strip('%'))  # Bandwidth %
        }
        nvidia_data.append(gpu_info)
    return nvidia_data


class BenchmarkRecord:
    """Class to store a single benchmark record with timing and GPU monitoring information."""

    def __init__(self, gpu_monitor_interval: float = 0.1):
        """Initialize an empty benchmark record.

        Args:
            gpu_monitor_interval (float): Interval in seconds between GPU measurements
        """
        self.start_time = None
        self.end_time = None
        self.inputs_len = 0
        self.outputs_len = 0
        self.gpu_monitor_thread = None
        self.gpu_monitor_active = False
        self.gpu_data = []
        # seconds between GPU measurements
        self.gpu_monitor_interval = gpu_monitor_interval

    def set_inputs_len(self, length: int):
        """Set the length of input tokens."""
        self.inputs_len = length

    def set_outputs_len(self, length: int):
        """Set the length of output tokens."""
        self.outputs_len = length

    def _gpu_monitor_worker(self):
        """Worker function that runs in a separate thread to monitor GPU usage."""
        while self.gpu_monitor_active:
            try:
                gpu_info = get_nvidia_smi_info()
                timestamp = time.time() - self.start_time
                self.gpu_data.append({
                    'timestamp': timestamp,
                    'gpu_info': gpu_info
                })
                time.sleep(self.gpu_monitor_interval)
            except Exception as e:
                print(f"Error in GPU monitoring: {e}")
                break

    def time_tick(self):
        """Record the start time and start GPU monitoring."""
        self.start_time = time.time()
        self.gpu_monitor_active = True
        self.gpu_data = []
        self.gpu_monitor_thread = threading.Thread(
            target=self._gpu_monitor_worker)
        self.gpu_monitor_thread.daemon = True
        self.gpu_monitor_thread.start()

    def time_tock(self):
        """Record the end time and stop GPU monitoring."""
        if self.start_time is not None:
            self.end_time = time.time()
        self.gpu_monitor_active = False
        if self.gpu_monitor_thread is not None:
            self.gpu_monitor_thread.join(timeout=1.0)

    @property
    def elapsed_time(self):
        """Calculate and return the elapsed time."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return 0

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Calculate and return GPU usage statistics."""
        if not self.gpu_data:
            return {}

        # Initialize stats dictionary
        stats = {}

        # Get all unique GPU indices
        gpu_indices = set()
        for data_point in self.gpu_data:
            for gpu in data_point['gpu_info']:
                gpu_indices.add(gpu['index'])

        # Calculate stats for each GPU
        for gpu_idx in gpu_indices:
            prefix = f"gpu_{gpu_idx}_"
            temps = []
            powers = []
            utils = []
            mem_used = []
            mem_utils = []

            for data_point in self.gpu_data:
                for gpu in data_point['gpu_info']:
                    if gpu['index'] == gpu_idx:
                        temps.append(gpu['temperature.gpu'])
                        powers.append(gpu['power.draw'])
                        utils.append(gpu['utilization.gpu'])
                        mem_used.append(gpu['memory.used'])
                        mem_utils.append(gpu['utilization.memory'])

            if temps:  # Only add stats if we have data for this GPU
                stats[f"{prefix}temp_avg"] = sum(temps) / len(temps)
                stats[f"{prefix}temp_max"] = max(temps)
                stats[f"{prefix}power_avg"] = sum(powers) / len(powers)
                stats[f"{prefix}power_max"] = max(powers)
                stats[f"{prefix}util_avg"] = sum(utils) / len(utils)
                stats[f"{prefix}util_max"] = max(utils)
                stats[f"{prefix}mem_used_avg"] = sum(mem_used) / len(mem_used)
                stats[f"{prefix}mem_used_max"] = max(mem_used)
                stats[f"{prefix}mem_util_avg"] = sum(
                    mem_utils) / len(mem_utils)
                stats[f"{prefix}mem_util_max"] = max(mem_utils)

        return stats

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        # Ensure timing is stopped if it was started
        if self.start_time is not None and self.end_time is None:
            self.time_tock()
        return False  # Don't suppress exceptions


class TimeBenchmarkRecorder:
    """Class to record and manage timing data for benchmarks."""

    def __init__(self, model_name: str, method_type: str, gpu_info_interval: float = 0.5):
        """
        Initialize the recorder.

        Args:
            model_name (str): Name of the model being benchmarked
            method_type (str): Type of method/implementation being benchmarked
            gpu_info_interval (float): Interval in seconds between GPU measurements
        """
        self.model_name = model_name
        self.method_type = method_type
        self.records = []
        self.gpu_info_interval = gpu_info_interval

    def new_record(self):
        """
        Create a new benchmark record to be used as a context manager.

        Returns:
            BenchmarkRecord: A new record object
        """
        return BenchmarkRecord(gpu_monitor_interval=self.gpu_info_interval)

    def append(self, record: BenchmarkRecord):
        """
        Append a completed record to the recorder.

        Args:
            record (BenchmarkRecord): The record to append
        """
        self.records.append(record)

    def get_stats(self):
        """
        Calculate statistics for all records.

        Returns:
            dict: Dictionary containing statistics (num_runs, avg_time, min_time, max_time, all_times)
        """
        if not self.records:
            return {
                "num_runs": 0,
                "avg_time": 0,
                "min_time": 0,
                "max_time": 0,
                "all_times": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "gpu_stats": {}
            }

        elapsed_times = [record.elapsed_time for record in self.records]
        input_tokens = sum(record.inputs_len for record in self.records)
        output_tokens = sum(record.outputs_len for record in self.records)

        # Collect GPU stats from all records
        gpu_stats = {}
        for record in self.records:
            record_gpu_stats = record.get_gpu_stats()
            if record_gpu_stats:
                # Average GPU stats across all runs
                for key, value in record_gpu_stats.items():
                    if key in gpu_stats:
                        # If we already have this stat, update it by averaging
                        current_count = gpu_stats[f"{key}_count"]
                        current_value = gpu_stats[key]
                        gpu_stats[key] = (
                            current_value * current_count + value) / (current_count + 1)
                        gpu_stats[f"{key}_count"] = current_count + 1
                    else:
                        # First time seeing this stat
                        gpu_stats[key] = value
                        gpu_stats[f"{key}_count"] = 1

        # Remove temporary count fields
        keys_to_remove = [
            key for key in gpu_stats.keys() if key.endswith("_count")]
        for key in keys_to_remove:
            del gpu_stats[key]

        return {
            "num_runs": len(self.records),
            "avg_time": sum(elapsed_times) / len(elapsed_times),
            "min_time": min(elapsed_times),
            "max_time": max(elapsed_times),
            "all_times": elapsed_times,
            # Average input tokens
            "input_tokens": input_tokens // len(self.records),
            # Average output tokens
            "output_tokens": output_tokens // len(self.records),
            "gpu_stats": gpu_stats
        }

    def clear(self):
        """Clear all records."""
        self.records = []


class TimeBenchmark:
    """Main class for benchmarking different model implementations."""
    method_type = "abstract"

    def __init__(
        self,
        model_name: str,
        data_path: str,
        context_max_length: int = 56000,
        device="cuda",
        max_data_count: int = None,
    ):
        """
        Initialize the benchmark.

        Requires:
        - model_name: The name of the model to benchmark
        - data_path: The path to the dataset file
            Currently; only longbench-v2 is supported
        - context_max_length:
            The maximum length of the context
            The context will be feed into a template.
            Therefore total length will slightly longer than this value.
        - device: The device to use for benchmarking
        - max_data_count: Maximum number of data points to use from the dataset
        """
        self.model_name = model_name
        self.data_path = data_path
        self.contex_max_length = context_max_length
        self.device = torch.device(device)
        self.max_data_count = max_data_count

        # Initialize model and tokenizer
        self.model, self.tokenizer = self.load_model_and_tokenizer(
            model_name, self.device)

    def load_model_and_tokenizer(self, model_name: str, device=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Initialize the model. This method should be implemented by subclasses.
        Requires:
        - model_name: 
        Returns:
        - model: The initialized model
        - tokenizer: The tokenizer corresponding to the model
        """
        raise NotImplementedError(
            "Subclasses must implement init_model method")

    def iter_data(self):
        """
        Iterate over the dataset.
        """

        with open(self.data_path, "r") as f:
            data = json.load(f)

        if self.max_data_count is not None:
            data = data[0:self.max_data_count]

        # Extract context from dataset - adapt this based on your dataset structure
        for item in tqdm(data):
            context = item["context"]
            # truncate
            _token = self.tokenizer(
                context,
                max_length=self.contex_max_length,
                truncation=True,
                return_tensors=None,
            )
            context = self.tokenizer.decode(
                _token.input_ids, skip_special_tokens=True)

            # Apply chat template
            inputs = self.tokenizer.apply_chat_template(
                [
                    [
                        {"role": "user", "content": f"The provided materials. {context} \n\nPlease write a summary of the material."},
                    ],
                ],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs = inputs.to(self.device)
            yield inputs.input_ids, inputs.attention_mask

    def run_benchmark(self, max_new_tokens=128, gpu_info_interval: Optional[float] = 1.0):
        """
        Run the benchmark for the specified number of runs.

        Args:
            max_new_tokens (int): Maximum number of new tokens to generate

        Returns:
            dict: Dictionary containing time and GPU statistics
        """
        recorder = TimeBenchmarkRecorder(
            self.model_name, self.method_type, gpu_info_interval=gpu_info_interval)

        for input_ids, attention_mask in self.iter_data():
            with recorder.new_record() as rec:
                rec.set_inputs_len(input_ids.shape[-1])
                rec.time_tick()
                output = self.infer(
                    input_ids,
                    attention_mask,
                    max_new_tokens,
                )
                rec.time_tock()
                rec.set_outputs_len(output.shape[-1])
                recorder.append(rec)

        # Return complete statistics including GPU data
        return recorder.get_stats()

    def infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ):
        """
        Perform inference. This method should be implemented by subclasses.

        Requires:
        - input_ids: The input token IDs
        - attention_mask: The attention mask
        - max_new_tokens: The maximum number of new tokens to generate
        """
        raise NotImplementedError("Subclasses must implement infer method")


class LRQKModelBenchmark(TimeBenchmark):
    """Benchmark implementation for LRQK model."""
    method_type = "LRQK"

    def __init__(
        self,
        lrqk_rank: int = 32,
        lrqk_num_active_tokens: int = 2048,
        lrqk_num_iter: int = 2,
        *args,
        **kwargs,
    ):
        self.lrqk_rank = lrqk_rank
        self.lrqk_num_active_tokens = lrqk_num_active_tokens
        self.lrqk_num_iter = lrqk_num_iter
        super().__init__(*args, **kwargs)

    def load_model_and_tokenizer(self, model_name: str, device=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize the LRQK model and tokenizer."""
        model, tokenizer = lrqk_attention.load_model(
            model_name,
            lrqk=True,
            device=device or self.device,
        )
        # Get model configuration
        _conf = model.config
        if hasattr(_conf, "num_key_value_heads"):
            num_key_value_groups = _conf.num_attention_heads // _conf.num_key_value_heads
        elif hasattr(_conf, "multi_query_group_num"):
            num_key_value_groups = _conf.multi_query_group_num

        self.num_key_value_groups = num_key_value_groups
        return model, tokenizer

    @torch.inference_mode()
    def infer(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int):
        """Perform inference with LRQK model."""

        # Generate with LRQK cache
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                past_key_values=lrqk_attention.DynamicLRQKCache(
                    num_key_value_groups=self.num_key_value_groups,
                    r=self.lrqk_rank,
                    num_active_tokens=self.lrqk_num_active_tokens,
                    lite_tokens=64,
                    max_iter=self.lrqk_num_iter,
                    tol=1e-8,
                    lwattn_factory=lrqk_attention.LightAttentionIndicesOffloadPrefill,
                ),
                do_sample=True,
            )
            return outputs[0]


class ShadowKVModelBenchmark(TimeBenchmark):
    """Benchmark implementation for ShadowKV model."""
    method_type = "ShadowKV"

    def load_model_and_tokenizer(self, model_name: str, device=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize the ShadowKV model and tokenizer."""
        assert "llama" in model_name.lower(), f"currently, only llama is supported"
        model = shadowkv_lib.Llama(model_name, attn_mode='shadowkv_cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    @torch.inference_mode()
    def infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ):
        """Perform inference with ShadowKV model."""
        _, outputs = self.model.generate(
            input_ids=input_ids,
            gen_len=max_new_tokens,
            benchmark=False,
            verbose=False,
            return_generated_ids=True,
        )
        return torch.LongTensor(outputs)


class VanillaModelBenchmark(TimeBenchmark):
    """Benchmark implementation for the original model."""
    method_type = "vanilla"

    def load_model_and_tokenizer(self, model_name: str, device=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Initialize the original model and tokenizer."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    @torch.inference_mode()
    def infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
    ):
        """Perform inference with the original model."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )
        return outputs

def get_benchmark(args) -> TimeBenchmark:
    """Get the benchmark implementation based on the specified method."""
    if args.method == "vanilla":
        return VanillaModelBenchmark(
            model_name=args.model_name,
            data_path=args.data_path,
            context_max_length=args.context_max_length,
            device=args.device,
            max_data_count=args.max_data_count,
        )
    elif args.method == "lrqk":
        return LRQKModelBenchmark(
            model_name=args.model_name,
            data_path=args.data_path,
            context_max_length=args.context_max_length,
            device=args.device,
            max_data_count=args.max_data_count,
            lrqk_rank=args.lrqk_rank,
            lrqk_num_active_tokens=args.lrqk_num_active_tokens,
            lrqk_num_iter=args.lrqk_num_iter,
        )
    elif args.method == "shadowkv":
        return ShadowKVModelBenchmark(
            model_name=args.model_name,
            data_path=args.data_path,
            context_max_length=args.context_max_length,
            device=args.device,
            max_data_count=args.max_data_count,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

def main():
    """Main function to run benchmarks with specified models and configurations."""
    parser = argparse.ArgumentParser(
        description="Benchmark different model implementations")

    # Add arguments
    parser.add_argument("--model_name", type=str, default="./llama_hf/Meta-Llama-3.1-8B-Instruct",
                        help="Name of the model to benchmark (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--data_path", type=str, default="./datasets/longbench-v2.json",
                        help="Path to the benchmark dataset file")
    parser.add_argument("--method", type=str, choices=["vanilla", "lrqk", "shadowkv"], required=True,
                        help="Which implementation to benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum number of new tokens to generate during benchmarking")
    parser.add_argument("--context_max_length", type=int, default=56000,
                        help="Maximum length of context tokens")
    parser.add_argument("--max_data_count", type=int, default=None,
                        help="Maximum number of data points to use from the dataset (default: use all data)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Directory path to save benchmark results (JSON format)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run benchmarks on (e.g., 'cuda', 'cpu')")
    parser.add_argument("--gpu_monitor_interval", type=float, default=0.5,
                        help="Interval in seconds between GPU monitoring measurements")

    # lrqk params
    parser.add_argument("--lrqk_rank", type=int, default=32,
                        help="Rank of the LRQK model")
    parser.add_argument("--lrqk_num_active_tokens", type=int, default=2048,
                        help="Number of active tokens for the LRQK model")
    parser.add_argument("--lrqk_num_iter", type=int, default=2,
                        help="Number of iterations for the LRQK model")

    # Parse arguments
    args = parser.parse_args()

    # Validate model name and method compatibility
    if args.method == "shadowkv" and "llama" not in args.model_name.lower():
        print("Warning: ShadowKV currently only supports LLaMA models.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting benchmark.")
            return

    if not args.method in ["vanilla", "lrqk", "shadowkv"]:
        print(f"Unknown method: {args.method}")
        return

    # Run benchmark
    print(
        f"Starting benchmark with {args.model_name} using {args.method} method...")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")
    print(f"Context max length: {args.context_max_length}")
    print(f"Max new tokens: {args.max_new_tokens}")
    if args.max_data_count is not None:
        print(f"Max data count: {args.max_data_count}")

    try:
        benchmark = get_benchmark(args)

        # Run benchmark
        results = benchmark.run_benchmark(
            max_new_tokens=args.max_new_tokens,
            gpu_info_interval=args.gpu_monitor_interval
        )

        # Print results
        print("Benchmark Results:")
        print(f"Number of runs: {results['num_runs']}")
        print(f"Average time: {results['avg_time']:.4f} seconds")
        print(f"Min time: {results['min_time']:.4f} seconds")
        print(f"Max time: {results['max_time']:.4f} seconds")
        print(f"Average input tokens: {results['input_tokens']}")
        print(f"Average output tokens: {results['output_tokens']}")

        # Print GPU stats if available
        if 'gpu_stats' in results and results['gpu_stats']:
            print("GPU Statistics:")
            for key, value in results['gpu_stats'].items():
                print(f"{key}: {value:.2f}")

        # Save results if output path specified
        if args.output_path:
            # Ensure output directory exists
            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)

            _extra_name = ""
            if args.method == "lrqk":
                _extra_name = f"_r{args.lrqk_rank}_act{args.lrqk_num_active_tokens}_{args.lrqk_num_iter}"

            # Generate a timestamp-based filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_safe_name = args.model_name.replace(
                '/', '_').replace('-', '_')
            results_path = os.path.join(
                args.output_path, f"{model_safe_name}_{args.method}{_extra_name}_{timestamp}.json")

            # Combine all results into a single file
            combined_results = {
                "model_name": args.model_name,
                "method": args.method,
                "context_max_length": args.context_max_length,
                "max_new_tokens": args.max_new_tokens,
                "device": args.device,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "benchmark_stats": results,
                "all_times": results["all_times"],
                "gpu_stats": results.get("gpu_stats", {}),
                "summary": {
                    "avg_time_sec": round(results["avg_time"], 4),
                    "min_time_sec": round(results["min_time"], 4),
                    "max_time_sec": round(results["max_time"], 4),
                    "input_tokens": results["input_tokens"],
                    "output_tokens": results["output_tokens"],
                    "tokens_per_second": round(results["output_tokens"] / results["avg_time"], 2)
                }
            }

            with open(results_path, 'w') as f:
                json.dump(combined_results, f, indent=2)
            print(f"All results saved to {results_path}")

    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
