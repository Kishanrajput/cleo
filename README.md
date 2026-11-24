# cleo

**C**ontinual **LE**arning t**O**olkit — generic, model-agnostic continual-learning building blocks (PyTorch / Lightning). No dependency on any particular dataset or experiment framework.

- `cleo.replay`: `ReplayBuffer` (quantile-stratified, growing), `task_inverse_weights`, `make_replay_loader`
- `cleo.regularizer`, `cleo.ewc`, `cleo.synaptic_intelligence`: quadratic-penalty regularizer, EWC Fisher importance, SI tracker (Lightning callback)
- `cleo.drift_detectors`: threshold, ADWIN, model-error, model-uncertainty, input/output OOD, debounced and combined detectors, plus `build_single_detector` / `build_combined_detector`
- `cleo.forgetting`: `evaluate_on_past_tasks` (injected metric), `forgetting_metrics`, `plot_forgetting_heatmap`
- `cleo.scan`: `scan_stream`, a data-agnostic "feed chunks to a detector until it fires" loop

## Install / test

```bash
uv venv && uv pip install -e ".[dev]"
pytest
```
`pip install cleo[plots]` adds matplotlib for the heatmap.
