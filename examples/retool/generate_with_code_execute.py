# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
import re
from typing import Any, Dict, List, Optional, Union
import json

try:
    from jinja2 import Template
except ImportError:
    raise ImportError("Jinja2 is required. Please install it with: pip install jinja2")

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample
from slime.utils.misc import print_eval

# Import reward models
try:
    from slime.rollout.rm_hub.math_dapo_utils import last_boxed_only_string, remove_boxed
except ImportError:
    raise ImportError("MathDapo is not installed")

# Import tool sandbox functionality
from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

# Jinja2 template for tool-enabled conversations
TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""


def format_conversation_with_tools(
    prompt: str, tools: List[Dict[str, Any]] = None, system_prompt: str = None, messages: List[Dict[str, Any]] = None
) -> str:
    """Format conversation using Jinja2 template with tool support"""
    template = Template(TOOL_TEMPLATE)

    # Prepare messages
    messages_to_render = []

    # Always add system message - use provided one or default
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = (
        "You are a program solution verifier that can use Python "
        "tools to verify whether a program is a correct solution to a coding problem. "
        "Use the code_interpreter tool when necessary to run any code needed for verification."
    )

    messages_to_render.append({"role": "system", "content": system_content})

    # Add user message if provided
    if prompt:
        messages_to_render.append({"role": "user", "content": prompt})

    # Add assistant responses from previous turns if provided
    if messages:
        messages_to_render.extend(messages)

    # Render template
    formatted_text = template.render(messages=messages_to_render, tools=tools or [])

    return formatted_text


def postprocess_predictions(prediction: str) -> tuple[Optional[str], Union[str, Dict[str, Any], List[tuple[str, Dict[str, Any]]]]]:
    """Extract actions and content (supports multiple <tool_call> blocks)"""
    # 1. Check for Answer:\boxed{...}
    answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    answer_match = re.search(answer_pattern, prediction, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    # 2. Check for one or more <tool_call> blocks
    tool_call_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    tool_call_matches = re.findall(tool_call_pattern, prediction, re.DOTALL)

    if tool_call_matches:
        results = []
        for json_str in tool_call_matches:
            try:
                json_str = json_str.replace("\n", "\\n")
                tool_call_data = json.loads(json_str)
                tool_name = tool_call_data.get("name")
                arguments = tool_call_data.get("arguments", {})

                if tool_name == "code_interpreter":
                    code = arguments.get("code", "").strip()
                    stdin_value = arguments.get("stdin", arguments.get("input", None))
                    if code:
                        results.append(("code", {"code": code, "stdin": stdin_value}))
            except (json.JSONDecodeError, KeyError, AttributeError):
                continue

        # If multiple tool calls were found, return all of them
        if results:
            return "multi_code", results
        # Otherwise, fall through

    # 3. <code>...</code>
    code_match = re.search(r"<code>(.*?)</code>", prediction, re.DOTALL)
    if code_match:
        return "code", {"code": code_match.group(1).strip(), "stdin": None}

    # 4. ```python ... ```
    python_code_match = re.search(r"```python\s*(.*?)\s*```", prediction, re.DOTALL)
    if python_code_match:
        return "code", {"code": python_code_match.group(1).strip(), "stdin": None}

    return None, ""


def postprocess_responses(resp: str) -> str:
    """Post-process response to ensure tag completeness"""

    # BOB: in qwen3-8b and above, the model generates thinkings and needs to get rid of it
    marker="</think>"
    if marker in resp:
        resp = resp.split(marker)[-1]

    # Handle <tool_call> tags (new format from Jinja2 template)
    if "<tool_call>" in resp:
        # Find the last occurrence of <tool_call>...</tool_call>
        tool_call_pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(tool_call_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle <code> tags
    if "</code>" in resp:
        return resp.split("</code>")[0] + "</code>"

    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*.*?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    # Handle Answer: \boxed{...} format (only format we need for math_dapo)
    if "Answer:" in resp and "\\boxed{" in resp:
        # Find the last occurrence of Answer: \boxed{...} with nested braces support
        answer_pattern = r"Answer:\s*\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
        matches = list(re.finditer(answer_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            return resp[: last_match.end()]

    return resp


async def execute_predictions(prediction: str, max_tools_calls_per_turn = 4) -> str:
    """Execute predictions and return results"""
    action, content = postprocess_predictions(prediction)

    if action == "code":
        # Content is already the Python code (extracted by
        # postprocess_predictions)
        code = content["code"].strip() if isinstance(content, dict) else str(content).strip()
        stdin_value = content.get("stdin") if isinstance(content, dict) else None
        if code:
            # TODO BOB: this will create a deadlock!!!
            async with SEMAPHORE:
                args = {"code": code}
                if stdin_value is not None:
                    args["stdin"] = stdin_value
                result = await tool_registry.execute_tool("code_interpreter", args)

            next_obs = f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
            done = False
        else:
            next_obs = "\n\n<interpreter>\nError: No Python code found" "\n</interpreter>\n\n"
            done = False
    elif action == "multi_code":
        # only execute up to max_tools_calls_per_turn
        results = []
        for i, (act, cont) in enumerate(content):
            if i >= max_tools_calls_per_turn:
                break
            if act == "code":
                code = cont["code"].strip() if isinstance(cont, dict) else str(cont).strip()
                stdin_value = cont.get("stdin") if isinstance(cont, dict) else None
                if code:
                    async with SEMAPHORE:
                        args = {"code": code}
                        if stdin_value is not None:
                            args["stdin"] = stdin_value
                        result = await tool_registry.execute_tool("code_interpreter", args)
                    results.append(f"<interpreter>\n{result}\n</interpreter>")
                else:
                    results.append("<interpreter>\nError: No Python code found\n</interpreter>")
        next_obs = "\n\n".join(results) + "\n\n"
        done = False
    elif action == "answer":
        next_obs = ""
        done = True
    else:
        next_obs = (
            "\nMy previous action is invalid. "
            "If I want to execute code, I should put the code between "
            "<code> and </code>. "
            "If I want to give the final answer, I should use the format "
            "'Answer: \\boxed{answer}'. Let me try again.\n"
        )
        done = False

    return next_obs, done


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Custom generation function supporting tool calls"""
    assert not args.partial_rollout, "Partial rollout is not supported for " "this function at the moment."
    assert sample is not None and isinstance(sample, Sample), "Sample must be provided and be an instance of Sample class."

    print_eval(f"=== New Sample Index {sample.index} ===")
    print_eval("starting generation...")
    prompt = sample.prompt

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Set up the initial prompt with system prompt and tools (outside the loop)
    tool_specs = tool_registry.get_tool_specs()
    prompt = format_conversation_with_tools(prompt=sample.prompt, tools=tool_specs)

    prompt_tokens_ids = state.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids = []
    loss_masks = []
    tool_call_count = 0  # Track actual tool call rounds
    output = None

    for turn in range(TOOL_CONFIGS["max_turns"]):
        print_eval(f"=== Turn {turn} ===")
        print_eval(f"response so far: {response}")

        # BOB: hardcoded otherwise https will complain and have no fallback and will cause program to abort
        ctx_len = 40959
        # if not ctx_len or ctx_len <= 0:
        #     ctx_len = getattr(state.tokenizer, "model_max_length", 40960)

        current_len = len(state.tokenizer(prompt + response, add_special_tokens=False)["input_ids"])
        allowed_new = max(0, ctx_len - current_len)

        current_sampling_params = sampling_params.copy()
        max_new = current_sampling_params.get("max_new_tokens", None)
        if max_new is None:
            current_sampling_params["max_new_tokens"] = allowed_new
        else:
            current_sampling_params["max_new_tokens"] = max(0, min(max_new, allowed_new))

        if current_sampling_params["max_new_tokens"] == 0:
            # No room to generate more tokens
            sample.status = Sample.Status.TRUNCATED
            print_eval("Context length limit reached, stopping generation.")
            break
            
        # Simple: just send prompt + response
        payload = {
            "text": prompt + response,
            # "sampling_params": sampling_params,
            "sampling_params": current_sampling_params,
        }

        # Log payload to wandb for debugging
        try:
            import wandb
            import weave

            if wandb.run is not None:
                # Count available tools (from tool_specs)
                available_tools = len(tool_specs)
                # Count tools used in the current response
                tools_used = response.count("<interpreter>")

                wandb.log(
                    {   
                        "debug/payload_length": len(prompt + response),
                        "debug/num_token": len(state.tokenizer(prompt + response)["input_ids"]),
                        "debug/available_tools": available_tools,
                        "debug/tools_used": tools_used,
                        "debug/turn": turn,
                    }
                )
        except ImportError:
            pass  # wandb not available

        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]
        cur_response = postprocess_responses(cur_response)
        print_eval(f"====== Post-processed cur_response ======: {cur_response}")

        # Record current response tokens
        cur_response_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]
        response += cur_response
        response_token_ids += cur_response_token_ids
        loss_masks += [1] * len(cur_response_token_ids)

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            print_eval("Length limit reached during generation.")
            break

        next_obs, done = await execute_predictions(cur_response, max_tools_calls_per_turn=TOOL_CONFIGS["max_tool_calls_per_turn"])
        print_eval(f"Next observation: {next_obs}")
        print_eval(f"Done: {done}")
        if done:
            break

        # Count tool calls (when we get interpreter output, it means a tool
        # was called)
        if "<interpreter>" in next_obs:
            tool_call_count += 1

        assert next_obs != "", "Next observation should not be empty."
        obs_tokens_ids = state.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
        print_eval(f"before obs, response is now: {response}")
        response += next_obs
        response_token_ids += obs_tokens_ids
        loss_masks += [0] * len(obs_tokens_ids)

        # Check if maximum tool call count reached
        # if tool_call_count >= TOOL_CONFIGS["max_tool_calls"]:
        #     break

    # Set sample attributes
    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_masks = loss_masks

    # Store payload information for wandb logging
    sample.payload_text = prompt + response
    sample.payload_has_system = "<|im_start|>system" in prompt + response
    sample.payload_has_tools = "# Tools" in prompt + response

    # Store tool call count for reward calculation
    sample.tool_call_count = tool_call_count
    sample.turn_finished = turn + 1

    if output is None:
        return sample

    # Set status
    match output["meta_info"]["finish_reason"]["type"]:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> Union[float, Dict[str, Any]]:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    # BOB: disable it for now because code execution may need longer context
    # solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    ground_truth = int(ground_truth)
    try:
        pred = int(remove_boxed(last_boxed_only_string(solution_str)))
    except Exception as e:
        pred = None
    correct = pred == ground_truth

    reward = 1.0 if correct else -1.0
    # acc = correct
    result = {
        "score": reward, # int
        "pred": pred, # int
        "gt": ground_truth, # int
    }

    return result

async def reward_func(args, sample, **kwargs):
    """Tool call reward function using math_dapo as primary reward model"""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    # Build complete solution string
    solution_str = sample.prompt + sample.response

    # Get ground truth answer - label is a string, not a dict
    ground_truth = sample.label if sample.label is not None else ""

    # use \\boxed{...} answer
    result = compute_score(solution_str, ground_truth, strict_box_verify=True)

    # WARNING: needs to check float or dict is the correct format
    return result
