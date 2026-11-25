# BESS Grid Control

Local battery control with SAC and rule-based policies in a three-phase OpenDSS network.

```powershell
git submodule update --init --recursive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use Python 3.11+. Run `python main.py` to simulate, `python train.py` to train or `python test.py` to evaluate.

[Experimental results](docs/sac-progress.md) · [Controllers](docs/controllers.md) · [Network](docs/network-diagram.md) · [Data](docs/data.md) · [Research plan](docs/research-plan.md)
