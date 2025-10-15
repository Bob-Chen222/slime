# Path to the script you want to call
target_script="/root/workspace/slime/examples/retool/retool_qwen3_14b_rl.sh"

for t in 8; do
    export MAX_TURNS=$t
    export MAX_TOOL_CALLS=$t
    echo ">>> Running with MAX_TURNS=$t MAX_TOOL_CALLS=$t"
    bash "$target_script"
done