import paddle
import os
import time
import numpy as np
import random
import platform
import traceback
import argparse
import sys
from pathlib import Path

from graph_net.paddle import utils
from graph_net import test_compiler_util


def set_seed(random_seed):
    paddle.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def get_hardware_name(device):
    if device == "cuda":
        return paddle.device.cuda.get_device_name(0)
    elif device == "cpu":
        return platform.processor()
    return "unknown"


def load_model(model_path):
    """动态加载模型"""
    from importlib.util import spec_from_loader, module_from_spec
    
    model_file = f"{model_path}/model.py"
    with open(model_file, "r") as f:
        code = f.read()
    
    module_name = Path(model_file).stem
    spec = spec_from_loader(module_name, loader=None)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    exec(compile(f"import paddle\n{code}", model_file, "exec"), module.__dict__)
    
    return module.GraphModule()


def get_input_dict(model_path):
    """获取输入字典 - 与test_compiler.py保持一致"""
    inputs_params = utils.load_converted_from_text(model_path)
    params = inputs_params["weight_info"]
    inputs = inputs_params["input_info"]
    params.update(inputs)
    return {k: utils.replay_tensor(v) for k, v in params.items()}


def get_input_spec(model_path):
    """获取输入规范 - 与test_compiler.py保持一致"""
    inputs_params_list = utils.load_converted_list_from_text(model_path)
    input_spec = [None] * len(inputs_params_list)
    for i, v in enumerate(inputs_params_list):
        dtype = v["info"]["dtype"]
        shape = v["info"]["shape"]
        input_spec[i] = paddle.static.InputSpec(shape, dtype)
    return input_spec


def get_compiled_model(model, compiler, model_path):
    """获取编译后的模型"""
    if compiler == "nope":
        return model
    input_spec = get_input_spec(model_path)
    build_strategy = paddle.static.BuildStrategy()
    compiled_model = paddle.jit.to_static(
        model,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )
    compiled_model.eval()
    return compiled_model


def measure_performance(model_call, synchronizer_func, warmup, trials):
    """测量性能 - 简化版本"""
    # Warmup runs
    for _ in range(warmup):
        model_call()
    synchronizer_func()

    # 性能测试
    e2e_times = []
    for i in range(trials):
        duration_box = test_compiler_util.DurationBox(-1)
        with test_compiler_util.naive_timer(duration_box, synchronizer_func):
            model_call()
        e2e_times.append(duration_box.value)
        print(f"Trial {i + 1}: e2e={duration_box.value:.4f} ms")

    return test_compiler_util.get_timing_stats(e2e_times)


def main():
    """命令行接口 - 简化的单模型测试"""
    parser = argparse.ArgumentParser(description="Test device performance with fixed random seeds")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for testing (e.g., 'cpu', 'cuda', or 'dcu')"
    )
    parser.add_argument(
        "--compiler",
        type=str,
        required=False,
        default="cinn",
        help="Compiler backend to use (cinn or nope)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        required=False,
        default=5,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--trials",
        type=int,
        required=False,
        default=10,
        help="Number of timing trials"
    )
    
    args = parser.parse_args()
    
    # 设置固定随机种子
    set_seed(123)
    
    # 打印基本配置信息
    print(f"[Config] device: {args.device}")
    print(f"[Config] compiler: {args.compiler}")
    print(f"[Config] hardware: {get_hardware_name(args.device)}")
    print(f"[Config] framework_version: {paddle.__version__}")
    print(f"[Config] warmup: {args.warmup}")
    print(f"[Config] trials: {args.trials}")

    # 运行测试
    success = False
    try:
        synchronizer_func = paddle.device.synchronize
        
        # 获取输入数据和模型
        input_dict = get_input_dict(args.model_path)
        model = load_model(args.model_path)
        model.eval()

        print(f"Run model with compiler: {args.compiler}")
        if args.compiler == "nope":
            compiled_model = model
        else:
            compiled_model = get_compiled_model(model, args.compiler, args.model_path)
        
        # 测量性能
        time_stats = measure_performance(
            lambda: compiled_model(**input_dict), 
            synchronizer_func, 
            args.warmup, 
            args.trials
        )
        success = True
        
        # 打印结果（不保存到文件，直接输出到控制台）
        print(f"[Result] model_path: {args.model_path}")
        print(f"[Result] compiler: {args.compiler}")
        print(f"[Result] device: {args.device}")
        print(f"[Result] e2e_mean: {time_stats['mean']:.5f}")
        print(f"[Result] e2e_std: {time_stats['std']:.5f}")
        
    except Exception as e:
        print(f"Run model failed: {str(e)}")
        print(traceback.format_exc())
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    main()