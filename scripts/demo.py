"""Live demo of the API, about 1 min 30: what a user sees, then a model swap without a restart, then a rollback.

    make demo-prep     # once, after `make pipeline`: adds v2 (the March retrain the trigger rejected), v1 stays champion
    make serve         # in a second terminal, before going on stage
    make demo-check    # 5 seconds: API up, v1 serving, v2 present, admin token set
    make demo          # the demo. Enter: run the step, s: skip it, q: quit. SHORT=1 skips the optional steps
    make demo-reset    # if a rehearsal stopped half-way: v1 champion again, API reloaded

Without make (Windows PowerShell): `python scripts/demo.py env | prep | check | run [--short] | reset`, see docs/demo.md.

Every command shown on screen is the command that runs (the admin token is read from .env and never printed).
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
WINDOWS = os.name == "nt"
if WINDOWS and COLOR:
    os.system("")  # turns on ANSI colours in the Windows console


def how(action: str) -> str:
    """The command to type for an action: the make target, or the script itself where there is no make."""
    if shutil.which("make"):
        return {"run": "make demo", "short": "make demo SHORT=1"}.get(action, f"make demo-{action}")
    return f"python scripts/demo.py {'run --short' if action == 'short' else action}"


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
    display: str            # what the room sees
    argv: list[str]         # what runs: the same command, without a shell (works the same on Windows)
    optional: bool = False


def curl(api: str, path: str, body: str | None = None, token: bool = False) -> tuple[str, list[str]]:
    method = ["-X", "POST"] if body is not None or path == "/admin/reload" else []
    display = f"curl -s {' '.join(method) + ' ' if method else ''}{api}{path}"
    argv = ["curl", "-s", *method, f"{api}{path}"]
    if body is not None:
        display += f" \\\n    -H 'content-type: application/json' \\\n    -d '{body}'"
        argv += ["-H", "content-type: application/json", "-d", body]
    if token:
        display += ' -H "X-Admin-Token: $ADMIN_TOKEN"'
        argv += ["-H", f"X-Admin-Token: {admin_token()}"]
    return display, argv


def python(*args: str) -> tuple[str, list[str]]:
    return "python -m " + " ".join(f'"{a}"' if " " in a else a for a in args), [sys.executable, "-m", *args]


def steps(port: int) -> list[Step]:
    api = f"localhost:{port}"
    predict = curl(api, "/predict", '{"zone_id": 237, "hour": "2020-02-20T18:00:00"}')
    out = [
        Step("is a model loaded, and which one?", *curl(api, "/health")),
        Step("which code and which data trained it?", *curl(api, "/model"), optional=True),
    ]
    if shutil.which("grep"):
        pattern = "path: data/processed/zone_hour.parquet"
        out.append(Step("...the same data hash DVC wrote in dvc.lock", f"grep -m1 -A2 '{pattern}' dvc.lock",
                        ["grep", "-m1", "-A2", pattern, "dvc.lock"], optional=True))
    out += [
        Step("a zone and an hour in, pickups out (and what really happened)", *predict),
        Step("a zone that does not exist: a clear error, not a crash",
             *curl(api, "/predict", '{"zone_id": 999, "hour": "2020-02-20T18:00:00"}')),
        Step(f"someone makes v{CHALLENGER} the champion in the registry",
             *python("taxi.registry", "promote", CHALLENGER, "--reason", "live demo")),
        Step("the API has not moved: nothing changes behind your back", *curl(api, "/health"), optional=True),
        Step("reloading without the token is refused", *curl(api, "/admin/reload"), optional=True),
        Step("with the token from .env (never on screen, never in Git): new model, no restart",
             *curl(api, "/admin/reload", token=True)),
        Step(f"same question, now answered by v{CHALLENGER}", *predict),
    ]
    if shutil.which("make"):
        out.append(Step("and back: one command, logged, the API reloads itself",
                        'make rollback REASON="end of demo"', ["make", "rollback", "REASON=end of demo"]))
    else:  # no make (Windows): the two commands `make rollback` runs
        out += [Step("and back: the champion label returns to the previous version, logged",
                     *python("taxi.registry", "rollback", "--reason", "end of demo")),
                Step("the API reloads it, still without a restart", *curl(api, "/admin/reload", token=True))]
    out.append(Step(f"v{REFERENCE} answers again", *predict, optional=True))
    return out


def run_step(step: Step, n: int, total: int, port: int) -> None:
    print(c(DIM, f"# {n}/{total} · {step.caption}"))
    print(c(BOLD, "$ ") + step.display)
    choice = input().strip().lower()
    if choice == "q":
        raise KeyboardInterrupt
    if choice == "s":
        print(c(DIM, "  (skipped)\n"))
        return
    is_curl = step.argv[0] == "curl"
    argv = step.argv + (["-w", "\n__HTTP__%{http_code}"] if is_curl else [])
    env = {**os.environ, "PORT": str(port)}
    for k in ("MAKELEVEL", "MAKEFLAGS", "MFLAGS"):  # `make rollback` behaves as if typed, not as a sub-make
        env.pop(k, None)
    try:
        r = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    except FileNotFoundError:
        print(c(RED, f"  {argv[0]} is not installed on this computer\n"))
        return
    out, code = r.stdout, None
    if is_curl:
        out, _, code = out.rpartition("__HTTP__")
        if code == "000":
            print(c(RED, f"  API not reachable on port {port}: is the server running?\n"))
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
    os.system("cls" if WINDOWS else "clear")
    print(c(BOLD, "NYC taxi demand · the API, live") + c(DIM, "   (Enter: run · s: skip · q: quit)\n"))
    try:
        for i, s in enumerate(todo, 1):
            run_step(s, i, len(todo), port)
    except (KeyboardInterrupt, EOFError):
        print(c(DIM, f"\n  demo stopped. `{how('reset')}` puts v1 back if needed."))


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
         "no .env: make setup (Windows: python scripts/demo.py env). Empty token: delete .env, "
         "run that again, then restart the server")
    champ = registry.champion()
    line(champ == REFERENCE, f"champion is v{REFERENCE} (found: v{champ})", how("reset"))
    versions = [str(v["version"]) for v in registry.status()["versions"]]
    line(CHALLENGER in versions, f"v{CHALLENGER} exists in the registry (versions: {', '.join(versions) or 'none'})",
         how("prep"))
    try:
        code, health = get(f"http://localhost:{port}/health")
        line(health.get("status") == "ok", f"API answers on port {port}: {health}", "restart the server")
        line(health.get("model_version") == champ, f"API serves the champion (v{health.get('model_version')})",
             how("reset"))
        code, pred = get(f"http://localhost:{port}/predict", "POST", {"zone_id": 237, "hour": "2020-02-20T18:00:00"})
        line(code == 200, f"zone 237, 20 Feb 18:00 → predicted {pred.get('predicted_pickups')}, "
                          f"actual {pred.get('actual_pickups')} (deck: 911.13 and 860)")
    except (urllib.error.URLError, OSError):
        line(False, f"API answers on port {port}", "start the server in another terminal (make serve)")
    print(c(GREEN, f"\nReady. `{how('short')}` when you are on stage.") if ok else c(RED, "\nNot ready yet: see FIX above."))
    return ok


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


def prep(port: int) -> None:
    from taxi import registry
    from taxi.config import data_path
    from taxi.monitor import monitor
    from taxi.retrain import retrain

    if not data_path().exists():
        sys.exit(f"no data at {data_path()}: run the pipeline first (make pipeline, or dvc repro)")
    if registry.champion() is None:
        sys.exit("no champion in the registry: run the pipeline first (make pipeline, or dvc repro -f train)")
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
    ap.add_argument("action", choices=["run", "check", "prep", "reset", "env"])
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT") or env_file().get("PORT") or 8000))
    ap.add_argument("--short", action="store_true", help="skip the optional steps (about 1 minute)")
    a = ap.parse_args()
    if a.action == "run":
        run(a.port, a.short)
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
