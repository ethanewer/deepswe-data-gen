# DeepSWE easiest-5 benchmark

Runs the DeepSWE easiest-5 subset (five fixed tasks) through Pier +
mini-swe-agent. `run.py` is the OpenAI-compatible runner used by `eval.run_all`
and works against any API (DeepSeek/OpenRouter) or a local server. The shell
scripts here automate the **LOCAL-GPU** setup that serves a model on the same
host with vLLM and runs the agent inside Pier Docker task containers.

## Subset definition

- `data/easiest_5_eval_split.json` is the task split read by `run.py`
  (`DEFAULT_TASK_SPLIT`). Do not move it.
- `data/leaderboard_analysis/` holds the leaderboard/solve-usage CSVs used to
  derive that split (provenance only; not read by any code).

## LOCAL-GPU run order

Run from the repo root with the `.venv-swe-uv` environment. `MODEL` is an env
override on the serve/eval scripts (they default to `Qwen/Qwen3.6-27B-FP8`).

1. `prepare_tasks.sh` — sparse-checkout the DeepSWE tasks to `TASKS_DIR`
   (`/tmp/deep-swe/tasks`) and verify the easiest-5 exist. (no GPU)
2. `patch_pier_proxy_safe_ports.sh` — add port 8000 to Pier's Squid
   `Safe_ports` so task containers can reach the host proxy. Idempotent. (no GPU)
3. `serve_vllm_replicas.sh` — start one vLLM replica per GPU (default GPUs 1-7
   at ports 8101-8107, leaving GPU 0 free). Use `RUN_IN_BACKGROUND=1`.
4. `serve_round_robin_proxy.sh` — start the round-robin proxy on `:8000`
   fanning across the replicas (`OPENAI_BACKENDS`). Task containers see it as
   `http://172.17.0.1.nip.io:8000/v1`.
5. `run_eval_local.sh` — launch the Pier/mini-swe-agent eval (n attempts per
   task). Writes the active job name to `/tmp/deepswe_active_job_name.txt`.
6. `monitor_job.sh` — poll the running job's `result.json` until it finishes.
7. `cleanup_eval.sh` — stop Pier jobs, the vLLM replicas, and the proxy, and
   remove DeepSWE task containers/networks (leaves unrelated GPU 0 work alone;
   override `PROCESS_REGEX` if you served a non-default `MODEL`).

Run outputs land under `runs/` (gitignored).

## API / non-local usage

To run against a hosted API instead of local serving, call `run.py` directly
(see the top-level `eval/README.md` "Individual Benchmarks" section), e.g.

```bash
.venv-swe-uv/bin/python -m eval.benchmarks.deepswe.run \
  --model deepseek-v4-flash \
  --litellm-model openai/deepseek-v4-flash \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY
```
