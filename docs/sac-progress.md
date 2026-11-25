# SAC progress

## Economic control

Across six 104-episode runs (discounts 0.99 and 0.999, seeds 42–44), the best saved policy in each run came from episode 1. All six exceeded the idle battery's adjusted node cost of 11,824.31. A causal tariff rule achieved 11,702.51 on the same twelve validation windows in 2023. Scores settle the difference between initial and final battery energy. The 2024 period is reserved for final testing.

Three-episode pilots tested balanced active power, SoC-aware action limits, multi-step returns, fixed entropy temperature and persistent exploration. None established an improvement over idle. At 50% SoC on two sampled mornings, an experimental critic preferred charging while its actor discharged. The outstanding question is whether the critic can select better actions over complete windows. [Research prompt and protocol](prompt-pesquisa-sac.md)

## Collection and training time

The trainer supports one or two OpenDSS collectors. Two processes advance separate training windows while one SAC learner updates from chronological replay buffers. Validation uses policy snapshots and logs adjusted costs after each trained episode. The MLP and GRU are available in the trainer.

The single-collector MLP reference (seed 44, discount 0.99) completed 104 episodes in **1 h 55 min 25 s**. The paired two-collector run stopped after **75 episodes and 75 validations**. Its projected completion time was about 1 h 15 min; a final wall-clock comparison is still open.

## CPU and CUDA pilots

Mean wall time on this machine with two OpenDSS collectors, 1,536 transitions, 1,281 updates and four abbreviated validations per run. Each configuration had two repetitions. Compare configurations within each experiment.

| Experiment | Configuration | Time |
|---|---|---:|
| MLP device comparison | CPU, four threads | 15.69 s |
| MLP device comparison | RTX 3050 OEM | 19.91 s |
| CUDA transfer changes | Before / after deferring metric copies and reducing batch transfers | 19.34 / 18.93 s |
| Adam on CUDA | Standard / fused | 18.92 / 18.50 s |
| LSTM, 64 units and eight steps | CPU / CUDA | 33.15 / 33.27 s |
| Same LSTM on CUDA | Contiguous recurrent weights on target and evaluation copies | 24.54 s |

A separate MLP benchmark with twelve abbreviated validations tested two, three and four collectors. CPU times were 20.90, 22.03 and 25.20 s; CUDA times were 22.97, 25.55 and 27.59 s. CPU with two collectors finished these short runs fastest. Four CPU collectors had the highest late throughput (111.9 steps/s). Three and four collectors, fused Adam and the LSTM were benchmark-only variants.

Local run artifacts live under `results/` (Git-ignored). The single-collector reference is `trial_20261123_173532_054558_gamma_0.99/seed_44`; the interrupted run is `full_collectors2_20261125_113118/seed_44`. Both are under `results/controllers/switzerland_5bus/sac/economic/node_cost/mlp/`. Use `initialization.json` for completed wall time and `training.jsonl` and `validation.jsonl` for episode records.
