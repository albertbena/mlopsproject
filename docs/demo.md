# Live demo of the API (Adrien, slide 5)

About **1 minute** (`make demo SHORT=1`, 7 steps) inside the 4:15–5:45 slot, or about **2 minutes** with the
optional steps (`make demo`, 12 steps). Everything runs locally: no network is needed once the pipeline has run.

The demo shows what slide 5 says, live: the API answers, refuses bad input, swaps the model **without a restart**
when the admin token is given, and goes back with `make rollback`. It ends with v1 serving, so nothing is left
changed for the rest of the defence.

## The day before (15 minutes, on the laptop that will be on the projector)

```bash
make setup          # .venv and a .env with an admin token
make pipeline       # needs internet: downloads the 6 TLC files (250 MB), trains v1 (~1 min)
make demo-prep      # adds v2 (the 2-12 March retrain the trigger rejected); v1 stays champion (~10 s)
make serve          # in a second terminal
make demo-check     # every line must say ok; zone 237 must show 911.13 / 860 as on the cover slide
make demo           # rehearse it, out loud, with a timer. Twice.
```

If a rehearsal stops half-way (`q`, or v2 still serving): `make demo-reset`.

## Ten minutes before the defence

1. Terminal 1: `make serve` (leave it running).
2. Terminal 2: `make demo-check`, then `make demo SHORT=1`. It clears the screen and waits on step 1: leave it there.
3. Font size 20+ (Cmd/Ctrl and `+`), light theme on a projector, terminal full screen, notifications off.
4. Check you can go from the deck to the terminal and back (Cmd/Alt+Tab) on the projector.

## On stage

Enter runs the step on screen, `s` skips it, `q` stops. The caption at the top of each step says what it proves,
so read it if your mind goes blank. Optional steps (only in the full version) are marked *opt*.

| Step | On screen | Say |
|---|---|---|
| intro | slide 5 | "Thanks Eduardo. I built the service that answers. Rather than a screenshot, here it is, running on my laptop." → terminal |
| 1 | `GET /health` → v1 | "First question: is a model loaded? Yes, version 1." |
| 2–3 *opt* | `GET /model`, then `dvc.lock` | "Which model exactly? Its git commit and its data hash. And here is dvc.lock: the same hash. From the API we can go back to the exact data." |
| 4 | `POST /predict` zone 237 | "A zone and an hour: zone 237, 20 February, 6 pm. We predict **911**, the real number was **860**." |
| 5 | zone 999 → 422 | "A zone that doesn't exist: a clear error, not a crash." |
| 6 | `promote 2` | "Now someone makes version 2 the champion in the registry. That's the retrain from early March." |
| 7 *opt* | `/health` → still v1 | "The registry changed, the API didn't: nothing moves behind your back." |
| 8 *opt* | reload, no token → 401 | "To switch, you need the token. Without it: refused." |
| 9 | reload with token → v2 | "The token lives in .env: never on screen, never in Git. Version 2, and I never restarted the server." |
| 10 | `/predict` → v2 | "Same question, new model, new answer." |
| 11 | `make rollback` → v1 | "And if it was a mistake: one command. The label goes back and the API reloads itself." |
| 12 *opt* | `/predict` → v1 | "911 again." |
| close | back to slide 5, dark card | "What I just did by hand, one of our **30 tests** does at every push on GitHub: promote, reload, roll back, reload, check who answers. Speaking of rollback, Rudy will explain how we choose which model answers." |

Do not run `make pipeline` on stage: it downloads 250 MB. Say "the pipeline takes about 30 seconds" and point at the
slide.

## On Windows (PowerShell, no make)

Windows PowerShell has no `make`, no `bash` and no `&&`: type one command per line. You need Python 3.11, 3.12 or
3.13 (`py --list`; otherwise `winget install Python.Python.3.13` and open a new PowerShell).

```powershell
git clone -b adri_lr/loving-ramanujan-7f61wo https://github.com/albertbena/mlopsproject.git
cd mlopsproject
py -3.13 -m venv .venv                          # or -3.12 / -3.11, whichever `py --list` shows
$env:Path = "$PWD\.venv\Scripts;$env:Path"     # in EVERY new PowerShell window: `python` is now the project's
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts\demo.py env                      # .env with a random admin token (what make setup does)
dvc repro                                       # what make pipeline does: needs internet, a few minutes
python scripts\demo.py prep                     # make demo-prep
```

Then, as above, with these commands instead of the make targets:

| make | PowerShell (after the `$env:Path` line, from the `mlopsproject` folder) |
|---|---|
| `make serve` | `uvicorn taxi.api.app:app --port 8000` |
| `make demo-check` | `python scripts\demo.py check` |
| `make demo SHORT=1` | `python scripts\demo.py run --short` |
| `make demo` | `python scripts\demo.py run` |
| `make demo-reset` | `python scripts\demo.py reset` |

Without make, `make rollback` is shown as the two commands it runs (registry rollback, then reload), so the short
demo has 8 steps instead of 7, and the dvc.lock step is left out (no `grep`).

## If something goes wrong

- **The API does not answer** (red line on screen): press `q`, go back to slide 5 and say "here is the same thing
  from our rehearsal": the slide's terminal shows the same requests and answers. Do not debug in front of the room.
- **A step fails after the promote**: carry on with the slides. Rudy's and Abdel's parts do not depend on this
  registry (the incident has its own database). Run `make demo-reset` after the defence.
- **Asked "how do I try it from a fresh clone?"**: `make setup`, `make pipeline`, `make serve`, then the `curl` of
  step 4. `make demo` runs the rest.
