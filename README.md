# fel-cl

Generic, model-agnostic continual-learning building blocks (PyTorch / Lightning). No dependency on any particular dataset or experiment framework.

- `fel_cl.replay`: `ReplayBuffer` (quantile-stratified, growing), `task_inverse_weights`, `make_replay_loader`
- `fel_cl.regularizer`, `fel_cl.ewc`, `fel_cl.synaptic_intelligence`: quadratic-penalty regularizer, EWC Fisher importance, SI tracker (Lightning callback)
- `fel_cl.drift_detectors`: threshold, ADWIN, model-error, model-uncertainty, input/output OOD, debounced and combined detectors, plus `build_single_detector` / `build_combined_detector`
- `fel_cl.forgetting`: `evaluate_on_past_tasks` (injected metric), `forgetting_metrics`, `plot_forgetting_heatmap`
- `fel_cl.scan`: `scan_stream`, a data-agnostic "feed chunks to a detector until it fires" loop

## Install / test

```bash
uv venv && uv pip install -e ".[dev]"
pytest
```
`pip install fel-cl[plots]` adds matplotlib for the heatmap.
