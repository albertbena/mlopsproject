"""Live demo of the whole project, about 2 minutes: the MLOps loop on the real data, from one terminal.

    1. which code and which data trained the model that serves (MLflow tags, the DVC hash)
    2. it works on 20 February; on 20 March, after COVID, the same model is far off
    3. the monitor sees the drift, the trigger retrains and promotes a better model
    4. the API swaps to it without a restart, and one command rolls it back

    make demo-prep    | python scripts/demo.py prep    once, after the pipeline (about 30 s)
    make serve        | uvicorn taxi.api.app:app       in a second terminal, before going on stage
    make demo-check   | python scripts/demo.py check   API up, v1 serving, the March example ready
    make demo         | python scripts/demo.py run     Enter: run the step, s: skip it, q: quit
    make demo-reset   | python scripts/demo.py reset   if a rehearsal stopped half-way: v1 champion again

`run --api` is the shorter demo of the API alone. Every command on screen is the command that runs (the admin
token is read from .env and never printed); the output of the monitor and of the trigger is cut to the lines that
matter, their full reports are saved under reports/.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

REFERENCE = "1"                      # the model trained on January-February, champion before the demo
CHALLENGER = "2"                     # `run --api` only: the 2-12 March retrain the trigger rejected
DRIFT_WEEK = "2020-03-09"            # the week that produces v2 (prep)
MARCH_WEEK = "2020-03-16"            # the week the demo monitors: demand -77%, the trigger promotes a new model
ZONE, FEB_HOUR = 237, "2020-02-20T18:00:00"
MARCH_DEFAULT = "2020-03-20T18:00:00"
EXAMPLE = ROOT / "reports" / "demo" / "march_example.json"   # the March hour chosen by prep (not committed)

# --------------------------------------------------------------------------------------------- display

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
WINDOWS = os.name == "nt"
if WINDOWS and COLOR:
    os.system("")  # turns on ANSI colours in the Windows console

DIM, BOLD, RED, GREEN, YELLOW, CYAN, MAGENTA = "2", "1", "1;31", "1;32", "1;33", "1;36", "1;35"
HIGHLIGHT = {  # the values the audience should look at
    "model_version": GREEN, "champion": GREEN, "serving": GREEN, "decision": GREEN, "action": YELLOW,
    "predicted_pickups": YELLOW, "actual_pickups": CYAN, "holdout_mae": YELLOW, "git_commit": MAGENTA,
    "train_period": CYAN, "detail": RED, "msg": RED,
}
KEY_VALUE = re.compile(r'("?)(\b[a-z_]+\b)("?\s*:\s*)("[^"]*"|[-0-9.]+|null|true|false)')
EMPHASIS: set[str] = set()  # strings coloured wherever they appear (the data hash)


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def how(action: str) -> str:
    """The command to type for an action: the make target, or the script itself where there is no make."""
    if shutil.which("make"):
        return {"run": "make demo", "serve": "make serve"}.get(action, f"make demo-{action}")
    if action == "serve":
        return ('$env:Path = "$PWD\\.venv\\Scripts;$env:Path" then uvicorn taxi.api.app:app --port 8000' if WINDOWS
                else ".venv/bin/uvicorn taxi.api.app:app --port 8000")
    return f"python scripts/demo.py {action}"


def paint(text: str) -> str:
    def repl(m: re.Match) -> str:
        q1, key, sep, value = m.groups()
        return f"{q1}{key}{sep}{c(HIGHLIGHT[key], value)}" if key in HIGHLIGHT else m.group(0)
    text = KEY_VALUE.sub(repl, text)
    for s in EMPHASIS:
        text = text.replace(s, c(MAGENTA, s))
    return text


def dump(obj, indent: int = 0) -> str:
    """JSON, indented, but with short lists and objects kept on one line (readable on a projector)."""
    flat = json.dumps(obj)
    if not isinstance(obj, (dict, list)) or len(flat) + indent <= 60 and indent:
        return flat
    pad = " " * (indent + 2)
    if isinstance(obj, dict):
        items = [f"{pad}{json.dumps(k)}: {dump(v, indent + 2)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [f"{pad}{dump(v, indent + 2)}" for v in obj]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def pretty(output: str) -> str:
    lines = []
    for line in output.rstrip().splitlines():
        try:
            obj = json.loads(line)
            line = dump(obj) if isinstance(obj, (dict, list)) else line
        except ValueError:
            pass
        lines.append(line)
    return paint("\n".join(lines))


def json_part(out: str) -> tuple[str, dict | None]:
    """Split a command's output into what comes before its JSON (e.g. the line make echoes) and the JSON."""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("{"):
            try:
                return "\n".join(lines[:i]), json.loads("\n".join(lines[i:]))
            except ValueError:
                break
    return out, None


def keep(*keys: str, drop: tuple[str, ...] = ()) -> Callable[[str], str]:
    """Show only some keys of a long JSON output (the full report is saved anyway)."""
    def show(out: str) -> str:
        head, obj = json_part(out)
        if obj is None:
            return out
        small = {k: obj[k] for k in keys if k in obj}
        for k in drop:  # "a.b" removes b inside a
            parent, _, child = k.partition(".")
            if isinstance(small.get(parent), dict):
                small[parent] = {kk: vv for kk, vv in small[parent].items() if kk != child}
        return (head + "\n" if head.strip() else "") + json.dumps(small)
    return show


# --------------------------------------------------------------------------------------------- settings

def env_file() -> dict:
    out = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def admin_token() -> str:
    return (os.environ.get("ADMIN_TOKEN") or env_file().get("ADMIN_TOKEN", "")).strip()


def get(url: str, method: str = "GET", body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    if token:
        req.add_header("X-Admin-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def march_example() -> dict:
    try:
        return json.loads(EXAMPLE.read_text())
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------------------------- the steps

@dataclass
class Step:
    caption: str
    display: str                                # what the room sees
    argv: list[str]                             # what runs: the same command, without a shell (same on Windows)
    optional: bool = False
    show: Callable[[str], str] | None = None    # cut a long output to what matters
    slow: bool = False                          # a few seconds: say so on screen
    kind: str = "raw"                           # how the web page draws the result (model, predict, monitor...)
    stage: str = ""                             # which part of the loop (trace, serve, drift, monitor, retrain...)
    who: str = ""                               # whose part of the project it shows


def curl(api: str, path: str, body: dict | None = None, token: bool = False) -> tuple[str, list[str]]:
    method = ["-X", "POST"] if body is not None or path == "/admin/reload" else []
    display = f"curl -s {' '.join(method) + ' ' if method else ''}{api}{path}"
    argv = ["curl", "-s", *method, f"{api}{path}"]
    if body is not None:
        data = json.dumps(body)
        display += f" \\\n    -H 'content-type: application/json' \\\n    -d '{data}'"
        argv += ["-H", "content-type: application/json", "-d", data]
    if token:
        display += ' -H "X-Admin-Token: $ADMIN_TOKEN"'
        argv += ["-H", f"X-Admin-Token: {admin_token()}"]
    return display, argv


def python(*args: str) -> tuple[str, list[str]]:
    return "python -m " + " ".join(f'"{a}"' if " " in a else a for a in args), [sys.executable, "-m", *args]


def make_or_python(target: list[str], module: list[str]) -> tuple[str, list[str]]:
    """`make <target>` where make exists, else the python command that target runs (Windows)."""
    if shutil.which("make"):
        return "make " + " ".join(target), ["make", *target]
    return python(*module)


def rollback_steps(api: str) -> list[Step]:
    if shutil.which("make"):
        return [Step("and if it goes wrong: one command, logged, the API reloads itself",
                     'make rollback REASON="end of demo"', ["make", "rollback", "REASON=end of demo"],
                     kind="rollback", stage="rollback", who="Rudy")]
    return [Step("and if it goes wrong: the champion label goes back to v1, logged",  # what `make rollback` runs
                 *python("taxi.registry", "rollback", "--reason", "end of demo"),
                 kind="rollback", stage="rollback", who="Rudy"),
            Step("the API reloads it, still without a restart", *curl(api, "/admin/reload", token=True),
                 kind="reload", stage="rollback", who="Adrien")]


def full_steps(port: int, data_hash: str | None, march: str) -> list[Step]:
    api = f"localhost:{port}"
    feb = curl(api, "/predict", {"zone_id": ZONE, "hour": FEB_HOUR})
    mar = curl(api, "/predict", {"zone_id": ZONE, "hour": march})
    report = f"reports/monitoring/{MARCH_WEEK}_v{REFERENCE}.json"
    steps = [Step("which model is serving, and which code and data trained it? (logged in MLflow)",
                  *curl(api, "/model"), kind="model", stage="trace", who="Eduardo")]
    if data_hash and (shutil.which("grep") or WINDOWS):
        cmd = ["grep", "-n", data_hash, "dvc.lock"] if shutil.which("grep") else ["findstr", "/n", data_hash, "dvc.lock"]
        steps.append(Step("the same data hash is in dvc.lock: DVC rebuilds exactly this data", " ".join(cmd), cmd,
                          kind="hash", stage="trace", who="Jules"))
    steps += [
        Step("it works: zone 237, 20 February 2020, 6 pm", *feb, kind="predict", stage="serve", who="Adrien"),
        Step("a bad request gets a clear error, not a crash",
             *curl(api, "/predict", {"zone_id": 999, "hour": FEB_HOUR}), optional=True,
             kind="error", stage="serve", who="Adrien"),
        Step("one month later, COVID: same zone, same time, same model", *mar, kind="predict", stage="drift"),
        Step("the monitor checks that week against v1's reference",
             *make_or_python(["monitor", f"START={MARCH_WEEK}"], ["taxi.monitor", "--start", MARCH_WEEK]),
             show=keep("status", "action", "reasons", "mean_pickups", "reference_mean_pickups"), slow=True,
             kind="monitor", stage="monitor", who="Abdel"),
        Step("the trigger trains a challenger on the last days, and promotes it only if it beats v1",
             *make_or_python(["retrain", f"REPORT={report}"], ["taxi.retrain", report]),
             show=keep("decision", "train", "holdout", "champion", "challenger", "serving", drop=("challenger.run_id",)),
             slow=True, kind="retrain", stage="retrain", who="Rudy"),
        Step("the API loads the new champion without a restart (admin token from .env, never on screen)",
             *curl(api, "/admin/reload", token=True), kind="reload", stage="swap", who="Adrien"),
        Step("same question as before, new model", *mar, kind="predict", stage="swap", who="Adrien"),
        *rollback_steps(api),
    ]
    return steps


def api_steps(port: int) -> list[Step]:
    """The shorter demo of the API alone (run --api): serve, refuse bad input, swap to v2, roll back."""
    api = f"localhost:{port}"
    predict = curl(api, "/predict", {"zone_id": ZONE, "hour": FEB_HOUR})
    a = {"who": "Adrien"}
    return [
        Step("is a model loaded, and which one?", *curl(api, "/health"), kind="health", stage="serve", **a),
        Step("which code and which data trained it?", *curl(api, "/model"), optional=True,
             kind="model", stage="trace", **a),
        Step("a zone and an hour in, pickups out (and what really happened)", *predict,
             kind="predict", stage="serve", **a),
        Step("a zone that does not exist: a clear error, not a crash",
             *curl(api, "/predict", {"zone_id": 999, "hour": FEB_HOUR}), kind="error", stage="serve", **a),
        Step(f"someone makes v{CHALLENGER} the champion in the registry",
             *python("taxi.registry", "promote", CHALLENGER, "--reason", "live demo"),
             kind="promote", stage="swap", who="Rudy"),
        Step("the API has not moved: nothing changes behind your back", *curl(api, "/health"), optional=True,
             kind="health", stage="swap", **a),
        Step("reloading without the token is refused", *curl(api, "/admin/reload"), optional=True,
             kind="error", stage="swap", **a),
        Step("with the token from .env (never on screen, never in Git): new model, no restart",
             *curl(api, "/admin/reload", token=True), kind="reload", stage="swap", **a),
        Step(f"same question, now answered by v{CHALLENGER}", *predict, kind="predict", stage="swap", **a),
        *rollback_steps(api),
    ]


# --------------------------------------------------------------------------------------------- running them

def json_docs(text: str) -> list:
    """Every JSON object in a command's output (make rollback prints two)."""
    dec, docs, i = json.JSONDecoder(), [], 0
    while (j := text.find("{", i)) >= 0:
        try:
            obj, i = dec.raw_decode(text, j)
            docs.append(obj)
        except ValueError:
            i = j + 1
    return docs


def execute(step: Step, port: int) -> dict:
    """Run a step's command and return what it printed (cut by `show`), its HTTP status for curl, and errors."""
    is_curl = step.argv[0] == "curl"
    argv = step.argv + (["-w", "\n__HTTP__%{http_code}"] if is_curl else [])
    env = {**os.environ, "PORT": str(port), "PYTHONIOENCODING": "utf-8"}
    for k in ("MAKELEVEL", "MAKEFLAGS", "MFLAGS"):  # `make ...` behaves as if typed, not as a sub-make
        env.pop(k, None)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {"ok": False, "error": f"{argv[0]} is not installed on this computer", "output": "", "docs": []}
    res = {"seconds": round(time.perf_counter() - t0, 2), "http": None, "error": None}
    out = r.stdout
    if is_curl:
        out, _, code = out.rpartition("__HTTP__")
        res["http"] = code.strip()
        if res["http"] == "000":
            res["error"] = f"API not reachable on port {port}: is the server running?"
    elif r.returncode:
        res["error"] = r.stderr.strip()[-600:] or f"exit code {r.returncode}"
    if step.show and not r.returncode:
        out = step.show(out)
    res["output"] = out.rstrip()
    res["docs"] = json_docs(out)
    res["ok"] = not res["error"] and (res["http"] is None or res["http"].startswith("2"))
    return res


def run_step(step: Step, n: int, total: int, port: int) -> None:
    print(c(DIM, f"# {n}/{total} · {step.caption}"))
    print(c(BOLD, "$ ") + step.display)
    choice = input().strip().lower()
    if choice == "q":
        raise KeyboardInterrupt
    if choice == "s":
        print(c(DIM, "  (skipped)\n"))
        return
    if step.slow:
        print(c(DIM, "  running..."), end="\r", flush=True)
    res = execute(step, port)
    if step.slow:
        print(" " * 20, end="\r")
    if res["http"] == "000" or not res["output"] and res["error"]:
        print(c(RED, f"  {res['error']}\n"))
        return
    if res["output"].strip():
        print(pretty(res["output"]))
    if res["error"] and res["http"] is None:
        print(c(RED, res["error"]))
    if res["http"]:
        print(c(GREEN if res["http"].startswith("2") else RED, f"HTTP {res['http']}"))
    print()


def run(port: int, short: bool, api_only: bool) -> None:
    if api_only:
        todo, title = api_steps(port), "NYC taxi demand · the API, live"
    else:
        try:
            data_hash = get(f"http://localhost:{port}/model")[1].get("data_md5")
        except (urllib.error.URLError, OSError):
            data_hash = None
        if data_hash:
            EMPHASIS.add(data_hash)
        todo = full_steps(port, data_hash, march_example().get("hour", MARCH_DEFAULT))
        title = "NYC taxi demand · the whole loop, live"
    todo = [s for s in todo if not (short and s.optional)]
    os.system("cls" if WINDOWS else "clear")
    print(c(BOLD, title) + c(DIM, "   (Enter: run · s: skip · q: quit)\n"))
    try:
        for i, s in enumerate(todo, 1):
            run_step(s, i, len(todo), port)
    except (KeyboardInterrupt, EOFError):
        print(c(DIM, f"\n  demo stopped. `{how('reset')}` puts v1 back if needed."))


# --------------------------------------------------------------------------------------------- before the demo

def check(port: int) -> bool:
    items = checks(port)
    print(c(BOLD, "Demo check"))
    for it in items:
        mark = c(GREEN, "ok  ") if it["good"] else (c(RED, "FIX ") if it["blocking"] else c(DIM, "--  "))
        print(f"  {mark} {it['text']}" + (c(DIM, f"   → {it['fix']}") if it["fix"] and not it["good"] else ""))
    ok = all(it["good"] or not it["blocking"] for it in items)
    print(c(GREEN, f"\nReady. `{how('run')}` when you are on stage.") if ok else c(RED, "\nNot ready yet: see FIX above."))
    return ok


def checks(port: int) -> list[dict]:
    """Everything the demo needs, as a list of {good, text, fix, blocking} (printed by check, shown by the page)."""
    from taxi import registry

    items: list[dict] = []

    def line(good: bool, text: str, fix: str = "", blocking: bool = True) -> None:
        items.append({"good": bool(good), "text": text, "fix": fix, "blocking": blocking})

    line(bool(admin_token()), "ADMIN_TOKEN is set in .env",
         "no .env: make setup (Windows: python scripts/demo.py env). Empty token: delete .env, "
         "run that again, then restart the server")
    champ = registry.champion()
    line(champ == REFERENCE, f"champion is v{REFERENCE} (found: v{champ})", how("reset"))
    ex = march_example()
    line(bool(ex), f"March example: zone {ex.get('zone_id', ZONE)}, {ex.get('hour', '?')} → v1 predicts "
                   f"{ex.get('v1', '?')}, the retrained model {ex.get('new', '?')}, actual {ex.get('actual', '?')}",
         how("prep"))
    versions = [str(v["version"]) for v in registry.status()["versions"]]
    line(CHALLENGER in versions, f"v{CHALLENGER} in the registry (only for `run --api`)", how("prep"), blocking=False)
    try:
        code, health = get(f"http://localhost:{port}/health")
        line(health.get("status") == "ok", f"API answers on port {port}: {health}", "restart the server")
        line(health.get("model_version") == champ, f"API serves the champion (v{health.get('model_version')})",
             how("reset"))
        code, pred = get(f"http://localhost:{port}/predict", "POST", {"zone_id": ZONE, "hour": FEB_HOUR})
        line(code == 200, f"zone 237, 20 Feb 18:00 → predicted {pred.get('predicted_pickups')}, "
                          f"actual {pred.get('actual_pickups')} (deck: 911.13 and 860)")
    except (urllib.error.URLError, OSError):
        line(False, f"API answers on port {port}", f"in another terminal: {how('serve')}")
    return items


def init_env() -> None:
    """What `make setup` does for .env: copy .env.example with a fresh random admin token (never overwrites)."""
    env = ROOT / ".env"
    if env.exists():
        print(".env already exists" + ("" if admin_token() else ": ADMIN_TOKEN is empty, delete .env and run this again"))
        return
    text = re.sub(r"(?m)^ADMIN_TOKEN=\s*$", f"ADMIN_TOKEN={secrets.token_urlsafe(32)}", (ROOT / ".env.example").read_text())
    env.write_text(text)
    print(".env created with a random admin token")


def reload_api(port: int) -> None:
    try:
        code, body = get(f"http://localhost:{port}/admin/reload", "POST", token=admin_token())
        print(f"  API reloaded: HTTP {code} {body}")
    except (urllib.error.URLError, OSError):
        print("  API not running: it will serve the champion when the server starts.")


def reset(port: int, reason: str = "reset after a demo rehearsal") -> None:
    from taxi import registry

    if registry.champion() != REFERENCE:
        print(json.dumps(registry.rollback(reason=reason, to=REFERENCE), indent=2))
    print(f"  champion: v{registry.champion()}, previous: v{registry.previous()}")
    reload_api(port)


def choose_march_example() -> dict:
    """Pick the March hour the demo asks about, so that the example on screen is a clear one.

    The trigger's challenger is deterministic (same data, same settings, fixed random_state), so it is trained here
    once without registering it, and compared with v1 on the zone's holdout hours (days it is not trained on).
    20 March 6 pm (one month after the February example) is kept when the difference is clear there.
    """
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    from taxi.config import params
    from taxi.features import between, feature_names
    from taxi.monitor import load_model
    from taxi.retrain import plan_dates
    from taxi.train import load_features, xy

    p = params()
    end = pd.Timestamp(MARCH_WEEK) + pd.Timedelta(days=p["monitoring"]["window_days"] - 1)
    dates = plan_dates(end.strftime("%Y-%m-%d"))
    feats = load_features()
    m = p["model"]
    new = HistGradientBoostingRegressor(loss=m["loss"], max_iter=m["max_iter"], learning_rate=m["learning_rate"],
                                        max_leaf_nodes=m["max_leaf_nodes"], random_state=m["random_state"])
    new.fit(*xy(between(feats, *dates["train"])))
    v1 = load_model(REFERENCE)[0]
    cols = feature_names(p["features"]["lags"], p["features"]["rolling_hours"])
    hold = between(feats, *dates["holdout"])
    hold = hold[hold["zone_id"] == ZONE].dropna(subset=cols).copy()
    X = hold[cols].astype("float64")
    hold["v1"], hold["new"] = v1.predict(X).clip(0), new.predict(X).clip(0)
    hold["gain"] = (hold["v1"] - hold["pickups"]).abs() - (hold["new"] - hold["pickups"]).abs()

    def clear(r) -> bool:
        return abs(r["new"] - r["pickups"]) <= 0.5 * abs(r["v1"] - r["pickups"]) and r["gain"] >= 5

    same_time = hold[hold["hour"] == pd.Timestamp(MARCH_DEFAULT)]
    if len(same_time) and clear(same_time.iloc[0]):
        row = same_time.iloc[0]
    else:
        busy = hold[hold["pickups"] >= hold["pickups"].median()]
        row = busy.loc[busy["gain"].idxmax()] if len(busy) else hold.loc[hold["gain"].idxmax()]
    return {"zone_id": ZONE, "hour": row["hour"].strftime("%Y-%m-%dT%H:%M:%S"), "actual": int(row["pickups"]),
            "v1": round(float(row["v1"]), 2), "new": round(float(row["new"]), 2),
            "holdout": list(dates["holdout"]), "train": list(dates["train"])}


def prep(port: int) -> None:
    from taxi import registry
    from taxi.config import data_path
    from taxi.monitor import monitor
    from taxi.retrain import retrain

    if not data_path().exists():
        sys.exit(f"no data at {data_path()}: run the pipeline first (make pipeline, or dvc repro)")
    if registry.champion() is None:
        sys.exit("no champion in the registry: run the pipeline first (make pipeline, or dvc repro -f train)")
    if registry.champion() != REFERENCE:
        registry.rollback(reason="demo prep: the demo starts from the reference model", to=REFERENCE)

    versions = [str(v["version"]) for v in registry.status()["versions"]]
    if CHALLENGER in versions:
        print(f"  v{CHALLENGER} already in the registry (used by `run --api`)")
    else:
        print(f"  v{CHALLENGER} for `run --api`: monitoring the week of {DRIFT_WEEK}, then the trigger...")
        report = monitor(DRIFT_WEEK)
        record = retrain(report, force=report["action"] != "retrain")
        print(f"  {record['decision']}: champion MAE {record['champion']['holdout_mae']}, "
              f"challenger v{record['challenger']['version']} MAE {record['challenger']['holdout_mae']}")
        if registry.champion() != REFERENCE:
            registry.rollback(reason="demo prep: the demo starts from the reference model", to=REFERENCE)

    print("  choosing the March example (the challenger is trained once here, not registered)...")
    ex = choose_march_example()
    EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLE.write_text(json.dumps(ex, indent=2) + "\n")
    print(f"  zone {ex['zone_id']}, {ex['hour']}: v1 predicts {ex['v1']}, the retrained model {ex['new']}, "
          f"actual {ex['actual']}")
    reset(port, reason=f"demo prep: v{REFERENCE} serves during the demo")


# --------------------------------------------------------------------------------------------- the web page

UI_PAGE = Path(__file__).with_name("demo_ui.html")


def serve_ui(port: int, ui_port: int, open_browser: bool) -> None:
    """A local web page that runs the same steps with buttons, for the projector (http://localhost:<ui_port>).

    The page only asks for "step i": the commands are the ones above, run here, one at a time. It listens on
    127.0.0.1 only, and never receives the admin token (only the command as displayed, and its output).
    """
    import contextlib
    import io
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    built: dict[str, list[Step]] = {}
    lock = threading.Lock()

    def build(mode: str) -> list[Step]:
        if mode == "api":
            steps = api_steps(port)
        else:
            try:
                data_hash = get(f"http://localhost:{port}/model")[1].get("data_md5")
            except (urllib.error.URLError, OSError):
                data_hash = None
            steps = full_steps(port, data_hash, march_example().get("hour", MARCH_DEFAULT))
        built[mode] = steps
        return steps

    def status() -> dict:
        from taxi import registry
        try:
            _, h = get(f"http://localhost:{port}/health")
            api = {"ok": h.get("status") == "ok", "version": h.get("model_version")}
        except (urllib.error.URLError, OSError, ValueError):
            api = {"ok": False, "version": None}
        return {"api": api, "champion": registry.champion(), "previous": registry.previous()}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body, ctype: str = "application/json") -> None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", ctype + "; charset=utf-8")
            self.send_header("cache-control", "no-store")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        def _route(self) -> tuple[str, str, dict]:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            return u.path, q.get("mode", ["full"])[0], q

        def do_GET(self) -> None:
            path_, mode, _ = self._route()
            if path_ in ("/", "/index.html"):
                return self._send(200, UI_PAGE.read_bytes(), "text/html")
            if path_ == "/api/steps":
                steps = build(mode)
                return self._send(200, {"mode": mode, "port": port, "example": march_example(), "steps": [
                    {"caption": s.caption, "display": s.display, "optional": s.optional, "slow": s.slow,
                     "kind": s.kind, "stage": s.stage, "who": s.who} for s in steps]})
            if path_ == "/api/status":
                return self._send(200, status())
            if path_ == "/api/check":
                items = checks(port)
                return self._send(200, {"items": items, "ready": all(i["good"] or not i["blocking"] for i in items)})
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            path_, mode, q = self._route()
            if path_ == "/api/run":
                steps = built.get(mode) or build(mode)
                i = int(q.get("i", ["-1"])[0])
                if not 0 <= i < len(steps):
                    return self._send(400, {"error": "no such step"})
                with lock:  # one command at a time, as in a terminal
                    res = execute(steps[i], port)
                return self._send(200, {"i": i, **res, "status": status()})
            if path_ == "/api/reset":
                buf = io.StringIO()
                with lock, contextlib.redirect_stdout(buf):
                    reset(port)
                return self._send(200, {"text": buf.getvalue(), "status": status()})
            self._send(404, {"error": "not found"})

    server = ThreadingHTTPServer(("127.0.0.1", ui_port), Handler)
    url = f"http://localhost:{ui_port}"
    print(c(BOLD, f"Demo page: {url}") + c(DIM, "   (Ctrl+C to stop it; the API must run in another terminal)"))
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n  demo page stopped")


def use_project_python() -> None:
    """Started with another Python (the venv line forgotten in this window)? Run again with the project's .venv."""
    venv = ROOT / ".venv"
    if Path(sys.prefix).resolve() == venv.resolve():
        return
    exe = venv / ("Scripts/python.exe" if WINDOWS else "bin/python")
    if exe.exists():
        sys.exit(subprocess.call([str(exe), str(Path(__file__).resolve()), *sys.argv[1:]]))
    sys.exit("the project is not installed: no .venv folder here. Run the installation first (docs/demo.md).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["ui", "run", "check", "prep", "reset", "env"])
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT") or env_file().get("PORT") or 8000))
    ap.add_argument("--short", action="store_true", help="skip the optional steps")
    ap.add_argument("--api", action="store_true", help="the shorter demo of the API alone")
    ap.add_argument("--ui-port", type=int, default=8501, help="port of the demo page (ui)")
    ap.add_argument("--no-browser", action="store_true", help="ui: do not open the browser")
    a = ap.parse_args()
    if a.action != "env":
        use_project_python()
    if a.action == "ui":
        serve_ui(a.port, a.ui_port, not a.no_browser)
    elif a.action == "run":
        run(a.port, a.short, a.api)
    elif a.action == "check":
        sys.exit(0 if check(a.port) else 1)
    elif a.action == "prep":
        prep(a.port)
    elif a.action == "env":
        init_env()
    else:
        reset(a.port)


if __name__ == "__main__":
    main()
