"""Retraining trigger: act on a monitoring report.

- action "retrain" (drift, clean data): train a challenger on the last `retrain.window_days` days of the window,
  keeping the last `retrain.holdout_days` days aside; score challenger and champion on that holdout; promote
  the challenger only if its MAE is lower. Both scores and the decision are saved.
- action "hold" (data quality problem) or "none": do nothing. `--force` overrides it (used in the incident
  report to show what retraining on a broken feed does).

    python -m taxi.retrain reports/monitoring/2020-03-16_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from taxi import registry, tracking
from taxi.config import data_path, params, path
from taxi.features import between
from taxi.monitor import load_model
from taxi.train import fit_and_register, load_features, xy


def plan_dates(window_end: str) -> dict:
    cfg = params()["retrain"]
    end = pd.Timestamp(window_end).normalize()
    day = lambda d: d.strftime("%Y-%m-%d")  # noqa: E731
    holdout_start = end - pd.Timedelta(days=cfg["holdout_days"] - 1)
    return {
        "train": (day(end - pd.Timedelta(days=cfg["window_days"] - 1)), day(holdout_start - pd.Timedelta(days=1))),
        "holdout": (day(holdout_start), day(end)),
    }


def retrain(report: dict, data: pd.DataFrame | None = None, force: bool = False, save: bool = True) -> dict:
    """Decide, and if needed train, compare and promote. Returns the decision record."""
    if report["action"] != "retrain" and not force:
        return {"decision": "no_retrain", "because": f"monitoring action is {report['action']!r}",
                "champion": registry.champion()}

    dates = plan_dates(report["window"]["end"])
    feats = load_features(data)
    data_md5 = tracking.frame_hash(data) if data is not None else tracking.file_md5(data_path())
    challenger = fit_and_register(
        feats, dates["train"], dates["holdout"], run_name=f"retrain-{dates['holdout'][1]}",
        tags={"trigger": "forced" if force and report["action"] != "retrain" else "drift",
              "monitoring_window": f"{report['window']['start']}..{report['window']['end']}"},
        data_md5=data_md5,
    )
    model, champ_version, _, _ = load_model()
    X, y = xy(between(feats, *dates["holdout"]))
    champ_mae = float(np.abs(model.predict(X) - y.to_numpy()).mean())
    chall_mae = challenger.metrics["valid_mae"]

    record = {
        "decision": "promote" if chall_mae < champ_mae else "keep_champion",
        "trigger": report["action"] if not force else f"forced (monitoring said {report['action']!r})",
        "reasons": report.get("reasons", []),
        "train": dates["train"], "holdout": dates["holdout"],
        "champion": {"version": champ_version, "holdout_mae": round(champ_mae, 4)},
        "challenger": {"version": challenger.version, "run_id": challenger.run_id, "holdout_mae": round(chall_mae, 4)},
    }
    if record["decision"] == "promote":
        registry.promote(challenger.version, reason=f"retrained after {record['trigger']}: holdout MAE "
                                                     f"{chall_mae:.2f} < champion {champ_mae:.2f}")
    record["serving"] = registry.champion()
    if save:
        out = path("reports") / "retrain" / f"{dates['holdout'][1]}_v{challenger.version}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="a JSON report written by taxi.monitor")
    ap.add_argument("--force", action="store_true", help="retrain even if the report says hold or none")
    ap.add_argument("--data", type=Path, default=None, help="another zone-hour Parquet file")
    a = ap.parse_args()
    data = pd.read_parquet(a.data) if a.data else None
    print(json.dumps(retrain(json.loads(a.report.read_text()), data, a.force), indent=2))


if __name__ == "__main__":
    main()
