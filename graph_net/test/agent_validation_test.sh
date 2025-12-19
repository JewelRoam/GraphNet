#!/usr/bin/env bash

GRAPH_NET_ROOT=$(python -c "import graph_net, os; print(os.path.dirname(os.path.dirname(graph_net.__file__)))")

OPTI_FILE_PATH="tmp/agent_gen/vit_base_patch32_clip_quickgelu_224.laion400m_e32_0/iter_0/model_new.py"

export TORCH_CUDA_ARCH_LIST="Ampere"
# optimized_interface means the class name in the optimized file to be tested
HANDLER_CONFIG=$(base64 -w 0 <<EOF
{
    "handler_path": "$GRAPH_NET_ROOT/graph_net/torch/sample_passes/agent_validate_handler.py",
    "handler_class_name": "AgentValidateGeneratorPass",
    "handler_config": {
        "optimized_file_path": "$OPTI_FILE_PATH", 
        "optimized_model_interface": "ModelNew",
        "device": "cuda",
        "warmup": 3,
        "trials": 10,
        "log_prompt": "Agent Validation"
    }
}
EOF
)

run_case() {
  local reference_model_path="$1"
  local name="$2"
  echo "[AgentTest] running $name sample at $reference_model_path"
  python -m graph_net.model_path_handler \
    --model-path "$reference_model_path" \
    --handler-config "$HANDLER_CONFIG"
}

run_case "tmp/agent_gen/vit_base_patch32_clip_quickgelu_224.laion400m_e32_0/iter_0/model.py" "Agent"

echo "[AgentTest] done."
