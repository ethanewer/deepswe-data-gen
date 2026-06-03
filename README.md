# DeepSWE Data Generation

Utilities for generating DeepSWE-related data files and smoke-testing the
selected SWE-rebench V2 task subset with the OpenAI API.

## Setup

```bash
python3 -m pip install -r requirements.txt
export OPENAI_API_KEY=...
```

## Regenerate Data

```bash
python3 scripts/run_data_generation.py
```

This regenerates the high-quality SWE-rebench V2 subset in `swerebench-v2/`.

## Smoke-Test DeepSWE

```bash
python3 scripts/run_deepswe.py --model gpt-5.4-mini --limit 1 --difficulty easy
```

Outputs are written to `runs/` as JSONL and are ignored by Git.

## Run Both

```bash
python3 scripts/run_all.py --model gpt-5.4-mini --limit 1 --difficulty easy
```
