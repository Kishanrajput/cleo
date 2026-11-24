# l3kit

**L3** (Life-**L**ong **L**earning) **Kit** — generic, model-agnostic continual-learning building blocks (PyTorch / Lightning). No dependency on any particular dataset or experiment framework.

- `l3kit.replay`: `ReplayBuffer` (quantile-stratified, growing), `task_inverse_weights`, `make_replay_loader`
- `l3kit.regularizer`, `l3kit.ewc`, `l3kit.synaptic_intelligence`: quadratic-penalty regularizer, EWC Fisher importance, SI tracker (Lightning callback)
- `l3kit.drift_detectors`: threshold, ADWIN, model-error, model-uncertainty, input/output OOD, debounced and combined detectors, plus `build_single_detector` / `build_combined_detector`
- `l3kit.forgetting`: `evaluate_on_past_tasks` (injected metric), `forgetting_metrics`, `plot_forgetting_heatmap`
- `l3kit.scan`: `scan_stream`, a data-agnostic "feed chunks to a detector until it fires" loop

## Install / test

```bash
uv venv && uv pip install -e ".[dev]"
pytest
```
`pip install l3kit[plots]` adds matplotlib for the heatmap.
