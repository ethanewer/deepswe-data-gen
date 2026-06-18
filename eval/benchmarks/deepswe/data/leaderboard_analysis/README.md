# DeepSWE Leaderboard Analysis (provenance only)

These CSVs are point-in-time exports from the DeepSWE open-source leaderboard and
the per-task solve-usage analysis used to derive the easiest-5 evaluation subset
(`../easiest_5_eval_split.json`, which is the only file in `data/` read by
`eval/benchmarks/deepswe/run.py`). They cover open-source model rankings, task
difficulty rankings (including a weak-models view), and per-task solve usage for
the easiest five tasks (overall, MiMo-only, and weak-vs-nonweak splits). They are
kept for provenance and are not consumed by any code.
