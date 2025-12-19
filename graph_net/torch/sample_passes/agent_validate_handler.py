import os
import sys
import torch
import traceback
from pathlib import Path
from collections import namedtuple

from graph_net import test_compiler_util
from graph_net.imp_util import load_module
from graph_net.torch.test_compiler import (
    measure_performance,
    get_hardward_name,
    compare_correctness,
)
from graph_net.sample_pass.sample_pass import SamplePass


EvalConfigDescriptor = namedtuple(
    "EvalConfigDescriptor",
    [
        "model_path",
        "device",
        "trials",
        "warmup",
        "log_prompt",
    ],
)


def print_with_log_prompt(key, value, log_prompt):
    print(f"{log_prompt} {key} {value}", file=sys.stderr, flush=True)


def print_basic_config(args, model_name, hardware_name, compile_framework_version):
    print_with_log_prompt("[Config] model:", model_name, args.log_prompt)

    print_with_log_prompt("[Config] device:", args.device, args.log_prompt)
    print_with_log_prompt("[Config] hardware:", hardware_name, args.log_prompt)
    print_with_log_prompt("[Config] warmup:", args.warmup, args.log_prompt)
    print_with_log_prompt("[Config] trials:", args.trials, args.log_prompt)
    print_with_log_prompt(
        "[Config] compile_framework_version:",
        compile_framework_version,
        args.log_prompt,
    )


def set_gpu_arch(arch_list: list[str]):
    """
    Set env variable for torch cuda arch list to build kernels for specified architectures
    """
    valid_archs = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada"]
    for arch in arch_list:
        if arch not in valid_archs:
            raise ValueError(
                f"Invalid architecture: {arch}. Must be one of {valid_archs}"
            )

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)


def validate_one_file(
    reference_model_path,
    optimized_model_path,
    optimized_model_interface,
    model_name,
    args,
    sync_class,
):
    ref_module = load_module(reference_model_path)
    opti_module = load_module(optimized_model_path)

    assert hasattr(
        opti_module, optimized_model_interface
    ), f"Optimized module does not have the interface: {optimized_model_interface}"

    print_basic_config(args, model_name, get_hardward_name(args), "")

    runtime_seed = 1024
    eager_failure = False
    expected_out = None
    eager_time_stats = {}

    torch.manual_seed(runtime_seed)
    model = getattr(ref_module, "Model")()
    # callback func, return list of args, already .to(device) for Tensor inputs
    get_inputs_func = getattr(ref_module, "get_inputs")

    try:

        def eager_model_call():
            return model(*get_inputs_func())

        torch.manual_seed(runtime_seed)
        expected_out, eager_time_stats = measure_performance(
            eager_model_call, args, sync_class
        )

        if not isinstance(expected_out, tuple):
            expected_out = (expected_out,)
    except (TypeError, RuntimeError) as e:
        print(f"Eager model execution failed: {str(e)}", file=sys.stderr)
        eager_failure = True

    compiled_failure = False
    optimized_model = None
    compiled_time_stats = {}

    try:
        optimized_model = getattr(opti_module, optimized_model_interface)()

        def optimized_model_call():
            return optimized_model(*get_inputs_func())

        torch.manual_seed(runtime_seed)
        compiled_out, compiled_time_stats = measure_performance(
            optimized_model_call, args, sync_class
        )

        if not isinstance(compiled_out, tuple):
            compiled_out = (compiled_out,)
    except (TypeError, RuntimeError) as e:
        print(f"Compiled model execution failed: {str(e)}", file=sys.stderr)
        compiled_failure = True
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print(f"debug-model-execution {type(e).__name__} {args.model_path}", flush=True)
    except Exception as e:
        compiled_failure = True
        print("\n--- Full Traceback ---")
        traceback.print_exc()
        print(f"debug-model-execution {type(e).__name__} {args.model_path}", flush=True)

    if eager_failure:
        print(f"{args.log_prompt} [Result] status: failed", file=sys.stderr, flush=True)
        print(
            f"{args.log_prompt} [Fail due to eager model execution error.]",
            file=sys.stderr,
            flush=True,
        )
    elif compiled_failure:
        print(f"{args.log_prompt} [Result] status: failed", file=sys.stderr, flush=True)
        print(
            f"{args.log_prompt} [Fail due to compiled model execution error.]",
            file=sys.stderr,
            flush=True,
        )
    else:
        compare_correctness(expected_out, compiled_out, args)

        print(
            f"{args.log_prompt} [Result] status: success", file=sys.stderr, flush=True
        )

        test_compiler_util.print_times_and_speedup(
            args, eager_time_stats, compiled_time_stats
        )


class AgentValidateGenerator:
    """Validates the optimized kernel against the reference implementation."""

    def __init__(
        self,
        model_name: str,
        reference_file_path: Path,
        optimized_file_path: Path,
        optimized_model_interface: str,
        args: EvalConfigDescriptor,
    ):
        self.model_name = model_name
        self.reference_file_path = reference_file_path
        self.optimized_file_path = optimized_file_path
        self.optimized_model_interface = optimized_model_interface
        self.args = args

    def validate(self):
        """Validate the optimized kernel against the reference implementation."""

        validate_one_file(
            self.reference_file_path,
            self.optimized_file_path,
            self.optimized_model_interface,
            self.model_name,
            self.args,
            self,
        )

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class AgentValidateGeneratorPass(SamplePass):
    def __init__(self, config=None):
        super().__init__(config)
        print(f"[AgentValidateGeneratorPass] {self.config=}")

    def declare_config(
        self,
        optimized_file_path: str,
        optimized_model_interface: str,
        device: str,
        # gpu_arch_list: list,
        warmup: int,
        trials: int,
        log_prompt: str,
    ):
        pass

    def __call__(self, reference_model_path: str):
        """Validate the optimized kernel against the reference implementation."""

        reference_model_path = Path(reference_model_path)
        optimized_model_path = Path(self.config["optimized_file_path"])

        if not reference_model_path.exists():
            raise FileNotFoundError(
                f"Reference model file not found: {reference_model_path}"
            )
        if not optimized_model_path.exists():
            raise FileNotFoundError(
                f"Optimized model file not found: {optimized_model_path}"
            )

        reference_model_name = reference_model_path.name.replace(".py", "")
        optimized_model_interface = self.config["optimized_model_interface"]

        # gpu_arch_list = self.config.get("gpu_arch_list", [])
        # if gpu_arch_list:
        #     set_gpu_arch(gpu_arch_list)
        # else:
        #     print("[Warning] gpu_arch_list is empty, may cause issues with CUDA kernel compilation.", flush=True)

        # To align with the test_compiler logs style.
        args = EvalConfigDescriptor(
            model_path=reference_model_path,
            device=self.config.get("device", "cuda"),
            trials=self.config["trials"],
            warmup=self.config["warmup"],
            log_prompt=self.config["log_prompt"],
        )

        validator = AgentValidateGenerator(
            model_name=reference_model_name,
            reference_file_path=reference_model_path,
            optimized_file_path=optimized_model_path,
            optimized_model_interface=optimized_model_interface,
            args=args,
        )
        validator.validate()
