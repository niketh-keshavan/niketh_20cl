"""File loading helpers."""
from pathlib import Path
import pandas as pd

def load_spike_csv(path: Path) -> pd.DataFrame:
    """Return a DataFrame with columns time_A_s, time_B_s."""
    df = pd.read_csv(path)
    expected = {"time_A_s", "time_B_s"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path.name} missing required columns {expected}")
    return df

def load_groups(group_map: dict[str, list[str]], datadir: Path) -> dict[str, list[pd.DataFrame]]:
    """Return {group: [DF, DF, …]}."""
    out: dict[str, list[pd.DataFrame]] = {}
    for group, names in group_map.items():
        out[group] = [load_spike_csv(datadir / n) for n in names]
    return out
