"""Sample data for Ford: 24 machines, 120 days of sensor readings, and 3 years of labelled failure history."""
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = [("Stamping press", 4), ("Welding robot", 8), ("Paint robot", 3), ("Conveyor drive", 5), ("CNC machining", 4)]
MODES = {"bearing": (1, 0.25, 0.1), "overheat": (0.2, 1, 0.3), "electrical": (0.1, 0.3, 1)}
# machine index, failure mode, day degradation starts, speed
DEGRADING = [(2, "bearing", 78, 0.17), (9, "overheat", 92, 0.12), (14, "bearing", 100, 0.1), (19, "electrical", 100, 0.17), (6, "electrical", 110, 0.1), (22, "overheat", 70, 0.05)]


def generate(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    machines = []
    for t, n in TYPES:
        for i in range(n):
            machines.append({"machine_id": f"{t.split()[0][:4].upper()}-{i + 1:02d}", "type": t})
    machines = pd.DataFrame(machines)
    machines["days_since_maintenance"] = rng.integers(10, 161, len(machines))
    readings = []
    for i, m in machines.iterrows():
        d = next((x for x in DEGRADING if x[0] == i), None)
        bv, bt, bc = rng.uniform(2, 4), rng.uniform(55, 70), rng.uniform(30, 60)
        w = MODES[d[1]] if d else (0, 0, 0)
        for day in range(120):
            k = max(0, day - d[2]) * d[3] * 1.15 if d else 0
            readings.append({"machine_id": m.machine_id, "day": day + 1, "vibration_mm_s": round(bv * (1 + rng.normal(0, 0.06) + 0.09 * k * w[0]), 3),
                             "temperature_c": round(bt * (1 + rng.normal(0, 0.03) + 0.032 * k * w[1]), 2), "motor_current_a": round(bc * (1 + rng.normal(0, 0.05) + 0.07 * k * w[2]), 2)})
    hist = []
    for i in range(2600):
        mode = rng.choice(list(MODES), p=[0.5, 0.3, 0.2]); w = MODES[mode]
        sev = rng.uniform(4, 10) if rng.random() < 0.18 else rng.uniform(0, 3.2)
        pm = int(rng.integers(5, 221))
        z = [rng.normal(sev * w[j], 0.8) for j in range(3)]
        fail = int(rng.random() < 1 / (1 + np.exp(-(-5 + 0.55 * sum(z) + 0.008 * pm))))
        hist.append({"window_id": i + 1, "vibration_z": round(z[0], 3), "temperature_z": round(z[1], 3), "current_z": round(z[2], 3),
                     "days_since_maintenance": pm, "failed_within_7d": fail})
    return {"machines": machines, "sensor_readings": pd.DataFrame(readings), "failure_history": pd.DataFrame(hist)}


# ---------------------------------------------------------------------------
# Save and load helpers
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def save(data: dict, folder: Path = DATA_DIR) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(folder / f"{name}.csv", index=False)


def load_or_generate(folder: Path = DATA_DIR, seed: int = 42) -> dict:
    """Read the CSVs in data/ if they exist, otherwise generate fresh sample data."""
    folder = Path(folder)
    names = list(generate(seed).keys())
    if all((folder / f"{n}.csv").exists() for n in names):
        return {n: pd.read_csv(folder / f"{n}.csv") for n in names}
    return generate(seed)


if __name__ == "__main__":
    d = generate(42)
    save(d)
    for n, df in d.items():
        print(f"data/{n}.csv  {len(df):,} rows x {df.shape[1]} columns")
