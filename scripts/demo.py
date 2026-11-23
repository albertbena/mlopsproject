"""Live demo of the API, about 1 min 30: what a user sees, then a model swap without a restart, then a rollback.

    make demo-prep     # once, after `make pipeline`: adds v2 (the March retrain the trigger rejected), v1 stays champion
    make serve         # in a second terminal, before going on stage
    make demo-check    # 5 seconds: API up, v1 serving, v2 present, admin token set
    make demo          # the demo. Enter: run the step, s: skip it, q: quit. SHORT=1 skips the optional steps
    make demo-reset    # if a rehearsal stopped half-way: v1 champion again, API reloaded

Every command shown on screen is the command that runs (the admin token is read from .env and never printed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
REFERENCE, CHALLENGER = "1", "2"
DRIFT_WEEK = "2020-03-09"  # the first drift alert: the trigger retrains on 2-12 March and keeps v1

# --------------------------------------------------------------------------------------------- display

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


DIM, BOLD, RED, GREEN, YELLOW, CYAN, MAGENTA = "2", "1", "1;31", "1;32", "1;33", "1;36", "1;35"
HIGHLIGHT = {  # the values the audience should look at
    "model_version": GREEN, "champion": GREEN, "previous": DIM,
    "predicted_pickups": YELLOW, "actual_pickups": CYAN, "data_md5": MAGENTA, "md5": MAGENTA,
    "git_commit": MAGENTA, "detail": RED, "msg": RED,
}
KEY_VALUE = re.compile(r'("?)(\b[a-z_]+\b)("?\s*:\s*)("[^"]*"|[-0-9.]+|null|true|false)')


def paint(text: str) -> str:
    def repl(m: re.Match) -> str:
        q1, key, sep, value = m.groups()
        return f"{q1}{key}{sep}{c(HIGHLIGHT[key], value)}" if key in HIGHLIGHT else m.group(0)
    return KEY_VALUE.sub(repl, text)


def dump(obj, indent: int = 0) -> str:
    """JSON, indented, but with short lists and objects kept on one line (readable on a projector)."""
    flat = json.dumps(obj)
    if not isinstance(obj, (dict, list)) or len(flat) + indent <= 60 and indent:
        return flat
    pad, items = " " * (indent + 2), []
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


# --------------------------------------------------------------------------------------------- the demo

@dataclass
class Step:
    caption: str
    command: str
    optional: bool = False


def steps(port: int) -> list[Step]:
    api = f"localhost:{port}"
    predict = (f"curl -s -X POST {api}/predict \\\n    -H 'content-type: application/json' \\\n"
               "    -d '{\"zone_id\": 237, \"hour\": \"2020-02-20T18:00:00\"}'")
    return [
        Step("is a model loaded, and which one?", f"curl -s {api}/health"),
        Step("which code and which data trained it?", f"curl -s {api}/model", optional=True),
        Step("...the same data hash DVC wrote in dvc.lock",
             "grep -m1 -A2 'path: data/processed/zone_hour.parquet' dvc.lock", optional=True),
        Step("a zone and an hour in, pickups out (and what really happened)", predict),
        Step("a zone that does not exist: a clear error, not a crash",
             f"curl -s -X POST {api}/predict \\\n    -H 'content-type: application/json' \\\n"
             "    -d '{\"zone_id\": 999, \"hour\": \"2020-02-20T18:00:00\"}'"),
        Step(f"someone makes v{CHALLENGER} the champion in the registry",
             f'python -m taxi.registry promote {CHALLENGER} --reason "live demo"'),
        Step("the API has not moved: nothing changes behind your back",
             f"curl -s {api}/health", optional=True),
        Step("reloading without the token is refused",
             f"curl -s -X POST {api}/admin/reload", optional=True),
        Step("with the token from .env (never on screen, never in Git): new model, no restart",
             f'curl -s -X POST {api}/admin/reload -H "X-Admin-Token: $ADMIN_TOKEN"'),
        Step(f"same question, now answered by v{CHALLENGER}", predict),
        Step("and back: one command, logged, the API reloads itself", 'make rollback REASON="end of demo"'),
        Step(f"v{REFERENCE} answers again", predict, optional=True),
    ]


def run_step(step: Step, n: int, total: int, port: int) -> None:
    print(c(DIM, f"# {n}/{total} · {step.caption}"))
    print(c(BOLD, "$ ") + step.command)
    choice = input().strip().lower()
    if choice == "q":
        raise KeyboardInterrupt
    if choice == "s":
        print(c(DIM, "  (skipped)\n"))
        return
    cmd = step.command.replace("python -m", f"{sys.executable} -m", 1) if step.command.startswith("python ") else step.command
    is_curl = cmd.startswith("curl ")
    if is_curl:
        cmd += " -w '\\n__HTTP__%{http_code}'"
    env = {**os.environ, "ADMIN_TOKEN": admin_token(), "PORT": str(port)}
    for k in ("MAKELEVEL", "MAKEFLAGS", "MFLAGS"):  # `make rollback` behaves as if typed, not as a sub-make
        env.pop(k, None)
    r = subprocess.run(cmd, shell=True, cwd=ROOT, env=env, capture_output=True, text=True)
    out = r.stdout
    code = None
    if is_curl:
        out, _, code = out.rpartition("__HTTP__")
        if code == "000":
            print(c(RED, f"  API not reachable on port {port}: is `make serve` running?\n"))
            return
    if out.strip():
        print(pretty(out))
    if r.returncode and not is_curl:
        print(c(RED, r.stderr.strip()[-600:]))
    if code:
        print(c(GREEN if code.startswith("2") else RED, f"HTTP {code}"))
    print()


def run(port: int, short: bool) -> None:
    todo = [s for s in steps(port) if not (short and s.optional)]
    os.system("clear" if os.name != "nt" else "cls")
    print(c(BOLD, "NYC taxi demand · the API, live") + c(DIM, "   (Enter: run · s: skip · q: quit)\n"))
    try:
        for i, s in enumerate(todo, 1):
            run_step(s, i, len(todo), port)
    except (KeyboardInterrupt, EOFError):
        print(c(DIM, "\n  demo stopped. `make demo-reset` puts v1 back if needed."))


# --------------------------------------------------------------------------------------------- before the demo

def check(port: int) -> bool:
    from taxi import registry

    ok = True

    def line(good: bool, text: str, fix: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {c(GREEN, 'ok  ') if good else c(RED, 'FIX ')} {text}" + (c(DIM, f"   → {fix}") if fix and not good else ""))

    print(c(BOLD, "Demo check"))
    line(bool(admin_token()), "ADMIN_TOKEN is set in .env",
         "no .env: make setup. Empty token: rm .env && make setup, then restart make serve")
    champ = registry.champion()
    line(champ == REFERENCE, f"champion is v{REFERENCE} (found: v{champ})", "make demo-reset")
    versions = [str(v["version"]) for v in registry.status()["versions"]]
    line(CHALLENGER in versions, f"v{CHALLENGER} exists in the registry (versions: {', '.join(versions) or 'none'})",
         "make demo-prep")
    try:
        code, health = get(f"http://localhost:{port}/health")
        line(health.get("status") == "ok", f"API answers on port {port}: {health}", "restart `make serve`")
        line(health.get("model_version") == champ, f"API serves the champion (v{health.get('model_version')})",
             "make demo-reset")
        code, pred = get(f"http://localhost:{port}/predict", "POST", {"zone_id": 237, "hour": "2020-02-20T18:00:00"})
        line(code == 200, f"zone 237, 20 Feb 18:00 → predicted {pred.get('predicted_pickups')}, "
                          f"actual {pred.get('actual_pickups')} (deck: 911.13 and 860)")
    except (urllib.error.URLError, OSError):
        line(False, f"API answers on port {port}", "run `make serve` in another terminal")
    print(c(GREEN, "\nReady. `make demo` when you are on stage.") if ok else c(RED, "\nNot ready yet: see FIX above."))
    return ok


def reload_api(port: int) -> None:
    try:
        code, body = get(f"http://localhost:{port}/admin/reload", "POST", token=admin_token())
        print(f"  API reloaded: HTTP {code} {body}")
    except (urllib.error.URLError, OSError):
        print("  API not running: it will serve the champion when `make serve` starts.")


def reset(port: int, reason: str = "reset after a demo rehearsal") -> None:
    from taxi import registry

    if registry.champion() != REFERENCE:
        print(json.dumps(registry.rollback(reason=reason, to=REFERENCE), indent=2))
    print(f"  champion: v{registry.champion()}, previous: v{registry.previous()}")
    reload_api(port)


def prep(port: int) -> None:
    from taxi import registry
    from taxi.config import data_path
    from taxi.monitor import monitor
    from taxi.retrain import retrain

    if not data_path().exists():
        sys.exit(f"no data at {data_path()}: run `make pipeline` first")
    if registry.champion() is None:
        sys.exit("no champion in the registry: run `make pipeline` (or `.venv/bin/dvc repro -f train`) first")
    versions = [str(v["version"]) for v in registry.status()["versions"]]
    if CHALLENGER in versions:
        print(f"  v{CHALLENGER} already in the registry, nothing to train")
    else:
        if registry.champion() != REFERENCE:
            registry.rollback(reason="demo prep: monitor the reference model", to=REFERENCE)
        print(f"  monitoring the week of {DRIFT_WEEK} with v{REFERENCE}...")
        report = monitor(DRIFT_WEEK)
        print(f"  action: {report['action']}, then the retraining trigger...")
        record = retrain(report, force=report["action"] != "retrain")
        print(f"  {record['decision']}: champion MAE {record['champion']['holdout_mae']}, "
              f"challenger v{record['challenger']['version']} MAE {record['challenger']['holdout_mae']}")
    reset(port, reason=f"demo prep: v{REFERENCE} serves during the demo")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "check", "prep", "reset"])
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT") or env_file().get("PORT") or 8000))
    ap.add_argument("--short", action="store_true", help="skip the optional steps (about 1 minute)")
    a = ap.parse_args()
    if a.action == "run":
        run(a.port, a.short)
    elif a.action == "check":
        sys.exit(0 if check(a.port) else 1)
    elif a.action == "prep":
        prep(a.port)
    else:
        reset(a.port)


if __name__ == "__main__":
    main()
