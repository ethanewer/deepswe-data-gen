# Local Serving Configs

These configs record the vLLM settings used for local benchmark runs. Start a
config with:

```bash
PYTHONPATH=. .venv/bin/python -m eval.model.serve_from_config \
  --config eval/serving/configs/qwen3_4b_thinking_2507_8gpu_65k.json \
  --background
```

The launcher starts one vLLM process per listed GPU, waits for backend health,
starts `eval.model.round_robin_proxy`, and writes a manifest under
`runs/serving/` with exact PIDs, commands, logs, and backend URLs.

For `Qwen/Qwen3-4B-Thinking-2507`, the config vendors the OpenThoughts-Agent
Qwen3 thinking chat template in `eval/chat_templates/qwen3_thinking_acc.jinja2`
and serves it with `--chat-template`.
