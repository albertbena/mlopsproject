# Live demo: the whole loop, in about 2 minutes

One terminal, the real data, every part of the project in the order it happens in production:

| Step | What runs | Whose part |
|---|---|---|
| 1–2 | `GET /model`, then the same data hash in `dvc.lock` | Eduardo (MLflow), Jules (DVC) |
| 3 | `POST /predict` zone 237, 20 Feb 2020, 6 pm: 911 predicted, 860 real | Adrien (API) |
| 4 | the same question on a March hour, after COVID: v1 is far off | (the drift) |
| 5 | the monitor on the week of 16 March: drift, action "retrain" | Abdel |
| 6 | the retraining trigger: trains a challenger live, compares, promotes it | Rudy |
| 7–8 | the API reloads the new champion without a restart; same March question, right again | Adrien |
| 9 | `make rollback`: v1 is back in one command | Rudy |

Everything runs locally: no network is needed once the pipeline has run. The demo ends with v1 serving again.
`make demo API=1` (or `run --api`) is the older, shorter demo of the API alone.

## Two ways to run it

- **The web page (recommended on a projector)**: `make demo-ui` (Windows: `python scripts\demo.py ui`) opens
  <http://localhost:8501> in the browser. One button per step (or `→`, `Space`, a presentation clicker), and each
  result is drawn instead of printed: the model card, the hash in `dvc.lock`, a taxi meter for each prediction, the
  monitor's verdict and the demand drop, the challenger against the champion, the version swap, the rollback. The top
  bar shows, live, which version the registry calls champion and which one the API serves. The command of each step
  is on screen, and "Raw output" shows exactly what it printed. Keys: `→`/`Space`/`Enter`/`PageDown` run the next
  step, `S` skips it, `←` shows the previous one. F11 for full screen.
- **The terminal**: `make demo` (Windows: `python scripts\demo.py run`), the same steps, Enter to run each one.

The page is served by `scripts/demo.py` on 127.0.0.1 only. It can only ask for "step n" of the list above, and the
admin token never reaches it.

## The day before (on the laptop that will be on the projector)

Linux or Mac:

```bash
make setup          # .venv and a .env with an admin token
make pipeline       # needs internet: downloads the 6 TLC files (250 MB), trains v1
make demo-prep      # about 30 s: v1 champion, chooses the March example (see below)
make serve          # in a second terminal
make demo-check     # every line ok; zone 237 must show 911.13 / 860 as on the cover slide
make demo-ui        # rehearse it, out loud, with a timer. Twice. (or `make demo` in the terminal)
```

Windows: see [On Windows](#on-windows-powershell-no-make) below, same steps with other commands.

If a rehearsal stops half-way (`q`, or a new model still serving): `make demo-reset`.

**The March example.** `demo-prep` trains the trigger's challenger once without registering it (it is deterministic:
same data, same settings, fixed seed, so the live retrain gives the same model) and keeps 20 March 6 pm when the
difference with v1 is clear there, otherwise the busiest holdout hour of zone 237 where it is clearest. It is an
illustration; the honest overall figure is the holdout MAE the trigger prints in step 6. `demo-check` shows the
numbers in advance.

**Version numbers.** Every run of the demo registers a new model (v3 the first time, then v4, v5...). Say "the new
model", not "v3".

## Ten minutes before

1. Terminal 1: the server (`make serve`), left running.
2. Terminal 2: `make demo-ui` (the browser opens on the ready check: every line ✓), F11 for full screen.
   Or, in the terminal: `make demo-check`, then `make demo`, which waits on step 1.
3. Notifications off. Zoom the browser (Ctrl +) if the room is large.

## On stage

Enter runs the step on screen, `s` skips it, `q` stops. The caption above each command says what it proves.
Steps 5 and 6 take a few seconds ("running..." on screen): keep talking. Read the March numbers from the screen.

| Step | Say |
|---|---|
| intro | "Let us show you the whole system, live, on the real data. Everything runs on this laptop." |
| 1 `/model` | "This is the model serving right now: version 1. MLflow kept the Git commit and the hash of the data that trained it." |
| 2 `dvc.lock` | "And here is dvc.lock, written by DVC: the same hash. We can rebuild exactly the data behind this model." |
| 3 February | "Does it work? Zone 237, 20 February, 6 pm: we predict 911, there were 860." |
| *opt* 422 | "A wrong request gets a clear error, not a crash." (skip it with `s` if short on time) |
| 4 March | "One month later, COVID. Same zone, same time, same model: it predicts *X*, there were *Y*. The world changed, the model did not." |
| 5 monitor | "Our monitor checks that week. Demand fell from 33 to 5 pickups an hour, the relative error is four times normal. Verdict: retrain." |
| 6 trigger | (while it runs) "The trigger trains a challenger on the last eleven days, keeps three days aside and tests both models on them." (result) "The challenger's error is 44% lower, so it is promoted. If it were worse, we would keep version 1: that happened the week before." |
| 7 reload | "The API loads the new model without a restart. It needs the admin token, which lives in .env: never on screen, never in Git." |
| 8 March again | "Same question, new model: *Z*, close to *Y* again." |
| 9 rollback | "And if the new model turns out bad: one command, logged, version 1 is back. That's the loop: which data, does it still work, how do we go back." |

Expected on the real data: monitor `PSI of pickups = 0.318`, `WAPE 54.1% is 4.09x the reference`, mean pickups 5.2
vs 33.5; trigger `train 2020-03-09..19`, `holdout 2020-03-20..22`, champion MAE 1.76, challenger 0.99, `promote`.

Do not run the pipeline on stage: it downloads 250 MB.

## On Windows (PowerShell, no make)

Windows PowerShell has no `make`, no `bash` and no `&&`: type one command per line. You need Python 3.11, 3.12 or
3.13 (`py --list`; otherwise `winget install Python.Python.3.13` and open a new PowerShell).

```powershell
git clone -b adri_lr/loving-ramanujan-7f61wo https://github.com/albertbena/mlopsproject.git
cd mlopsproject
py -3.13 -m venv .venv                          # or -3.12 / -3.11, whichever `py --list` shows
$env:Path = "$PWD\.venv\Scripts;$env:Path"     # in EVERY new PowerShell window
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts\demo.py env                      # .env with a random admin token (what make setup does)
dvc repro                                       # what make pipeline does: needs internet, a few minutes
python scripts\demo.py prep                     # make demo-prep
```

| make | PowerShell (after the `$env:Path` line, from the `mlopsproject` folder) |
|---|---|
| `make serve` | `uvicorn taxi.api.app:app --port 8000` |
| `make demo-check` | `python scripts\demo.py check` |
| `make demo-ui` | `python scripts\demo.py ui` |
| `make demo` | `python scripts\demo.py run` |
| `make demo-reset` | `python scripts\demo.py reset` |

Without make, each step shows the Python command its make target runs (`python -m taxi.monitor ...`), and the
rollback shows its two commands (registry rollback, then reload), so the demo has one step more. The hash step uses
`findstr` instead of `grep`.

## If something goes wrong

- **The API does not answer** (red line on screen): press `q` and go back to the slides. Do not debug in front of
  the room.
- **The demo stopped half-way**: carry on with the slides, and run `make demo-reset` afterwards.
- **Asked "how do I try it from a fresh clone?"**: `make setup`, `make pipeline`, `make serve`, then `make demo`.
