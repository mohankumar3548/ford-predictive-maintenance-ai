"""Machine health index, remaining life, failure prediction and maintenance scheduling for Ford."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import bar, kpi, line, money, num, pct, rec, table

META = json.loads(Path(__file__).with_name("meta.json").read_text())
PARAMS = [
    {"name": "downtime_cost_k", "label": "Unplanned downtime cost ($K per hour)", "min": 20, "max": 300, "step": 10, "default": 120},
    {"name": "buffer_days", "label": "Safety buffer before predicted failure (days)", "min": 0, "max": 10, "step": 1, "default": 3},
]
FAIL_HI = 8.0
SENSORS = ["vibration_mm_s", "temperature_c", "motor_current_a"]
SIGNAL = ["Vibration rising: bearing or alignment wear", "Temperature rising: cooling or lubrication fault", "Current rising: motor or electrical fault"]
REPAIR_HRS = [6, 4, 3]


def run(data, downtime_cost_k=120, buffer_days=3):
    mach, R, H = data["machines"], data["sensor_readings"], data["failure_history"]
    hx = H[["vibration_z", "temperature_z", "current_z", "days_since_maintenance"]]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(hx, H.failed_within_7d)
    auc = roc_auc_score(H.failed_within_7d, clf.predict_proba(hx)[:, 1])
    now = int(R.day.max())
    rows, curves = [], {}
    for m in mach.itertuples():
        g = R[R.machine_id == m.machine_id].sort_values("day")
        base = g[g.day <= 30][SENSORS]
        z = (g[SENSORS] - base.mean()) / base.std()
        hi = np.maximum(0, z.sum(axis=1) / 1.4).rolling(5, min_periods=1).mean().to_numpy()
        curves[m.machine_id] = hi
        last_z = z.tail(5).mean().to_numpy()
        slope = np.polyfit(np.arange(21), hi[-21:], 1)[0]
        rul = max(0.0, (FAIL_HI - hi[-1]) / slope) if slope > 0.02 else np.inf
        p7 = clf.predict_proba(pd.DataFrame([[*last_z, m.days_since_maintenance]], columns=hx.columns))[0, 1]
        dom = int(np.argmax(last_z))
        rows.append({"machine_id": m.machine_id, "type": m.type, "health_index": round(hi[-1], 2), "failure_prob_7d": round(p7, 3),
                     "days_to_failure": round(rul, 0) if np.isfinite(rul) else None, "signal": SIGNAL[dom] if hi[-1] > 2 else "Normal", "repair_hours": REPAIR_HRS[dom], "rul_raw": rul})
    M = pd.DataFrame(rows).sort_values("failure_prob_7d", ascending=False)

    fails = H[(H.failed_within_7d == 1) & (H[["vibration_z", "temperature_z", "current_z"]].max(axis=1) > 2.5)]
    km = KMeans(n_clusters=3, n_init=10, random_state=5).fit(fails[["vibration_z", "temperature_z", "current_z"]])
    pat = pd.DataFrame({"cluster": km.labels_}).value_counts().rename("failures").reset_index()
    names = ["Bearing and alignment wear", "Overheating", "Electrical and motor faults"]
    pat["pattern"] = [names[int(np.argmax(km.cluster_centers_[c]))] for c in pat.cluster]
    pat["share"] = (pat.failures / pat.failures.sum()).round(3)
    pat = pat[["pattern", "failures", "share"]].sort_values("share", ascending=False)

    windows = [(d, 8, "Sunday shutdown") if d % 7 == 6 else (d, 4, "Midweek gap") for d in range(now + 1, now + 61) if d % 7 in (2, 6)]
    jobs = []
    for r in M[(M.failure_prob_7d > 0.25) | (M.rul_raw < 45)].itertuples():
        due = now + (r.rul_raw if np.isfinite(r.rul_raw) else 60) - buffer_days
        w = next((w for w in reversed(windows) if w[0] <= due and w[1] >= r.repair_hours), None)
        jobs.append({"machine_id": r.machine_id, "type": r.type, "failure_prob_7d": r.failure_prob_7d, "days_to_failure": r.days_to_failure,
                     "signal": r.signal, "repair_hours": r.repair_hours, "book_into": f"Day {w[0] - now}, {w[2]}" if w else "Stop now"})
    J = pd.DataFrame(jobs, columns=["machine_id", "type", "failure_prob_7d", "days_to_failure", "signal", "repair_hours", "book_into"])
    unplanned = float((J.repair_hours * 3 + 6).sum()); planned_loss = float(J[J.book_into == "Stop now"].repair_hours.sum())
    saved_hrs = unplanned - planned_loss; saving = saved_hrs * downtime_cost_k * 1000
    top3 = M.machine_id.head(3).tolist()
    hc = pd.DataFrame({"day": np.arange(1, now + 1)});
    for mid in top3: hc[mid] = curves[mid].round(2)
    hc["failure_level"] = FAIL_HI
    finite = M[np.isfinite(M.rul_raw)].sort_values("rul_raw")
    M_out = M.drop(columns="rul_raw")
    return {
        "steps": [f"{len(mach)} machines x {now} days, {len(H):,} historical windows", "3 sensor z-scores per machine-day vs its first 30 days",
                  f"{int((M.health_index > 3).sum())} machines with health index above 3",
                  f"Shortest remaining life: {finite.iloc[0].machine_id} at {finite.iloc[0].rul_raw:.0f} days" if len(finite) else "No machine is degrading",
                  f"Model AUC {auc:.3f}; top risk {M.iloc[0].machine_id} at {pct(M.iloc[0].failure_prob_7d, 0)}",
                  ", ".join(f"{p.pattern} {pct(p.share, 0)}" for p in pat.itertuples()), f"{len(J)} jobs booked, {int((J.book_into == 'Stop now').sum())} needing an immediate stop", "Ready"],
        "kpis": [kpi("Machines at risk", str(len(J)), f"of {len(mach)} on the line"), kpi("Highest 7-day risk", pct(M.iloc[0].failure_prob_7d, 0), f"{M.iloc[0].machine_id}, {M.iloc[0].type}"),
                 kpi("Downtime avoided", f"{num(saved_hrs)} hrs", "Unplanned stops turned into planned work"), kpi("Saving", money(saving), f"At ${downtime_cost_k}K per downtime hour")],
        "charts": [line("Health index, three riskiest machines", hc, "day", top3 + ["failure_level"], "Combined sensor drift vs each machine's healthy baseline."),
                   bar("7-day failure probability (top 10)", M.head(10).assign(machine=lambda d: d.machine_id + " " + d.type), "machine", "failure_prob_7d"),
                   bar("How machines have failed (K-means on past failures)", pat, "pattern", "share")],
        "tables": [table("Maintenance schedule", J), table("Machine health board", M_out)],
        "recs": [rec(f"{j.machine_id}: {j.signal.split(': ')[-1]}", f"{j.type} at {pct(j.failure_prob_7d, 0)} risk of failing within 7 days. " + ("No planned window comes early enough; stop at the next shift change." if j.book_into == "Stop now" else f"Book the {j.repair_hours}-hour job into {j.book_into.lower()}."), j.book_into) for j in J.head(4).itertuples()]
                + [rec(f"Target {pat.iloc[0].pattern.lower()} across the line", f"{pct(pat.iloc[0].share, 0)} of past failures follow this pattern. Add checks for it to every Sunday shutdown.", "Prevent repeats")],
    }


if __name__ == "__main__":
    from src.data_generator import load_or_generate
    from src.report import write_reports

    out = run(load_or_generate(), **{p["name"]: p["default"] for p in PARAMS})
    for k in out["kpis"]:
        print(f"{k['label']}: {k['value']}  {k['note']}")
    print("Reports written to", write_reports(out, META, Path(__file__).resolve().parent.parent / "reports"))
