"""Simulated incident, end to end and reproducible: inject drift, let the system notice, react.

It runs in its own MLflow database (mlflow_incident.db), so it never touches the registry you serve from, and
writes every step to reports/incident/. docs/incident_report.md tells the story with these numbers.

Timeline (2020, real TLC data, plus one injected fault):
  1. reference model v1 trained on January-February, promoted to champion
  2. control week   2020-02-23: monitoring must say "ok"
  3. drift week     2020-03-09: demand starts to fall (COVID-19), monitoring alerts, retraining trigger runs
  4. next week      2020-03-16: same loop with the new champion
  5. injected fault 2020-03-23: the busiest zones go silent (a broken feed); monitoring must say "hold"
     and the trigger must refuse to retrain
  6. human override: someone forces a retrain on the broken week anyway
  7. the feed is fixed; the model trained on broken data is rolled back to the last clean one

    python -m taxi.incident
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from taxi.config import ROOT, data_path, params, path

INCIDENT_DB = ROOT / "mlflow_incident.db"
OUT = path("reports") / "incident"
SILENT_ZONES_COUNT = 3
FAULT_WEEK = ("2020-03-23", "2020-03-29")


def inject_silent_zones(df: pd.DataFrame, zones: list[int], start: str, end: str) -> pd.DataFrame:
    """A copy of the data where `zones` have 0 pickups from `start` to `end` (days included): a dead feed."""
    out = df.copy()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    mask = out["zone_id"].isin(zones) & (out["hour"] >= lo) & (out["hour"] < hi)
    out.loc[mask, "pickups"] = 0
    return out


def busiest_zones(df: pd.DataFrame, start: str, end: str, n: int) -> list[int]:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    week = df[(df["hour"] >= lo) & (df["hour"] < hi)]
    return [int(z) for z in week.groupby("zone_id")["pickups"].sum().nlargest(n).index]


def main() -> None:
    # Isolated tracking database and registry log, rebuilt from scratch on every run.
    for f in (INCIDENT_DB,):
        f.unlink(missing_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.json*"):
        f.unlink()
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{INCIDENT_DB}"
    os.environ["REGISTRY_LOG"] = str(OUT / "registry_log.jsonl")

    from taxi import registry, tracking
    from taxi.monitor import monitor
    from taxi.retrain import retrain
    from taxi.train import fit_and_register, load_features

    timeline = []

    def step(name: str, what: str, result: dict) -> dict:
        record = {"step": len(timeline) + 1, "name": name, "what": what, "result": result}
        timeline.append(record)
        (OUT / f"{record['step']:02d}_{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        r = result
        line = r.get("status") or r.get("decision") or r.get("action")
        print(f"[{record['step']:02d}] {name}: {line} {r.get('reasons', '')}")
        return result

    started = time.time()
    clean = pd.read_parquet(data_path())
    p = params()["periods"]

    ref = fit_and_register(load_features(clean), tuple(p["train"]), tuple(p["valid"]), run_name="reference",
                           tags={"trigger": "reference"}, data_md5=tracking.file_md5(data_path()))
    registry.promote(ref.version, reason="first model, trained on the reference period")
    step("reference", "Train v1 on the reference period and serve it",
         {"action": "promote", "version": ref.version, **{k: round(v, 4) for k, v in ref.metrics.items()}})

    step("control_week", "Monitor a normal week (sanity check)", monitor("2020-02-23", data=clean, save=False))

    r = step("drift_detected", "Monitor the week demand starts to fall", monitor("2020-03-09", data=clean, save=False))
    step("retrain", "Retraining trigger on that report", retrain(r, data=clean, save=False))

    r = step("next_week", "Monitor the next week with the current champion", monitor("2020-03-16", data=clean, save=False))
    step("retrain_again", "Retraining trigger on that report", retrain(r, data=clean, save=False))
    last_clean = registry.champion()

    zones = busiest_zones(clean, *FAULT_WEEK, SILENT_ZONES_COUNT)
    broken = inject_silent_zones(clean, zones, *FAULT_WEEK)
    r = step("fault_injected", f"Zones {zones} go silent from {FAULT_WEEK[0]} to {FAULT_WEEK[1]} (injected)",
             {**monitor(FAULT_WEEK[0], data=broken, save=False), "injected_silent_zones": zones})
    step("trigger_refuses", "Retraining trigger on that report", retrain(r, data=broken, save=False))

    forced = step("forced_retrain", "Human override: retrain on the broken week anyway",
                  retrain(r, data=broken, force=True, save=False))
    bad = forced["challenger"]["version"]
    if registry.champion() != bad:  # the holdout is broken too, so the bad model usually "wins": make sure it serves
        registry.promote(bad, reason="human override: promoted although monitoring said hold")
    step("bad_model_serving", "The model trained on the broken feed is now the champion",
         {"action": "serving", "champion": registry.champion(), "previous": registry.previous()})

    week = "2020-03-30"
    before = monitor(week, data=clean, save=False)
    clean_model = monitor(week, version=last_clean, data=clean, save=False)
    step("feed_fixed", f"Feed fixed: compare v{bad} and v{last_clean} on the clean week of {week}",
         {"action": "compare", "bad_model": {k: before[k] for k in ("model_version", "status", "mae", "wape", "wape_ratio")},
          "last_clean_model": {k: clean_model[k] for k in ("model_version", "status", "mae", "wape", "wape_ratio")}})

    t0 = time.time()
    rb = registry.rollback(reason=f"v{bad} was trained on a feed that failed the data quality check "
                                  f"({FAULT_WEEK[0]}..{FAULT_WEEK[1]}); back to v{last_clean}", to=last_clean)
    rb["seconds"] = round(time.time() - t0, 3)
    step("rollback", "Roll back to the last model trained on clean data", rb)

    after = monitor(week, data=clean, save=False)
    step("after_rollback", f"Monitor {week} with the champion after the rollback", after)

    summary = {"seconds": round(time.time() - started, 1), "silent_zones": zones, "bad_version": bad,
               "restored_version": last_clean, "final": registry.status()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "final"}, indent=2))


if __name__ == "__main__":
    main()
