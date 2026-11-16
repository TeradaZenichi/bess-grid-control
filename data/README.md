# Data

- `switzerland/5bus/`: active network, demand and device configuration.
- `switzerland/prices/standard_prices.csv`: active electricity tariff.
- `EV/ev_sessions.csv`: EV sessions consumed by the environment.
- `EV/evse_connections.csv`: companion connection matrix aligned with demand timestamps.
- `EV/support/`: generators, metadata and extraction reports.
- `old/`: other datasets, networks, tariffs and historical configurations.

Run the active scenario with `python main.py` from the repository root.
For data regeneration, see [EV support](EV/support/README.md).
