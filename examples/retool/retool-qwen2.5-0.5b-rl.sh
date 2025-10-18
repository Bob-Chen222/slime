#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/root/workspace/slime/scripts/models/qwen2.5-0.5B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/workspace/Qwen-weights/Qwen2.5-0.5B-weights
   --ref-load /root/workspace/Qwen-weights/Qwen2.5-0.5B-torch-dist
   --save /root/workspace/Qwen-weights/Qwen2.5-0.5B-iter
   --save-interval 4
   # --rotary-base 1000000
)

ROLLOUT_ARGS=(
   --prompt-data /root/workspace/synthetic-APPS-10.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 5
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.7

   --global-batch-size 4
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data apps /root/workspace/synthetic-APPS-emh30.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 4096
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

PPO_ARGS=(
    --advantage-estimator ppo
    --eps-clip 0.2
    --eps-clip-high 0.28
    --gamma 0.99
    --lambd 0.95
    --normalize-advantages
    --critic-lr 5e-6
    --entropy-coef 0.01
    --use-kl-loss
    --kl-loss-coef 0.1
    --kl-loss-type low_var_kl
)


OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-qwen2.5
   --wandb-group qwen2.5-0.5B-test-multi-turn
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_code_execute.generate
   --custom-rm-path generate_with_code_execute.reward_func
   --save-eval-debug-rollout-data /root/workspace/logs/eval_{rollout_id}.pt
   --exclude-truncated-samples
)

DEBUG_ARGS=(
   --debug-rollout-only
   --save-debug-rollout-data /root/workspace/logs/rollout_{rollout_id}.pt
)


# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} 
   # ${GRPO_ARGS[@]} \
   # ${DEBUG_ARGS[@]}