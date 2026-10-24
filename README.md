# Breathing Room

[![tests](https://github.com/Marc-Dvci/Breathing-Room/actions/workflows/ci.yml/badge.svg)](https://github.com/Marc-Dvci/Breathing-Room/actions/workflows/ci.yml)

Breathing Room turns a venue's camera video into three numbers per zone: how full it is
(people per m²), whether the crowd has started to push (a motion-based pushing index), and how
many minutes remain before the zone reaches a dangerous density at the current inflow. When a
zone needs attention, an agent checks the camera, counts the crowd a second way, finds where in
the zone people are pushing, compares the actions the venue can take, and puts one proposal in
front of a steward: a sentence to read over the radio, a camera crop of the spot, and the
predicted effect. The steward approves or rejects it, and 90 seconds later the system measures
whether the action worked. After the evening, a safety officer reads a report generated from the
tamper-evident record.

**Live console:** <https://44-199-44-59.sslip.io> (public demonstration, no login; production mode needs a steward to sign in; it pauses when nobody is watching and
starts a fresh evening for the next visitor) · **Film (4:35):** <https://youtu.be/zFVNj1GJoLQ> ·
**Technical report:** [docs/REPORT.md](docs/REPORT.md) · **Evening report:**
<https://44-199-44-59.sslip.io/report> · **MCP endpoint:** `https://44-199-44-59.sslip.io/mcp` (read-only)

Built for the OpenCV AI Competition 2026 by Marc Donovici, on OpenCV 5.0 and AWS.

![Steward console](docs/figures/console.png)

## Why a head count is not enough

At Itaewon, Seoul, on 29 October 2022, the first of 11 emergency calls warning of a crush came
at 18:34. Police investigators later estimated, from security footage and forensic simulation,
8 people per m² in the alley at about 22:15 and 9 to 11 per m² ten minutes later; 159 people
died. Since then, Seoul has put crowd-counting software on hundreds of CCTV cameras. A count says
how full a street is. It cannot tell a packed but calm crowd from one where people have started
to push, and it cannot say how much time is left. Helbing, Johansson and Al-Abideen, who analysed
the 2006 Jamarat crush frame by frame, proposed to quantify the danger of a crowd "in contrast to
purely density-based assessments" by its pressure: density times the local variance of
velocities. That is the index Breathing Room computes from video.

Breathing Room adds both, and measures them on real data:

| Claim | Result | Evidence |
|---|---|---|
| The pushing index, read from video alone, picks out the people that trained raters saw pushing | AUROC **0.798** [0.767, 0.828] over 9,260 rated person-seconds (342 people, 6 Jülich entrance runs); no training. The console's own code (`CameraZone`, at the half resolution it runs at) scores **0.797** on the same rows | [eval/test_a_pushing.py](eval/test_a_pushing.py), [eval/test_a_console.py](eval/test_a_console.py), [results](eval/results/test_a_scores.json) |
| It still works at equal density, where a head count carries no signal | AUROC **0.86, 0.84, 0.74, 0.73, 0.69** within the 0–2, 2–3, 3–4, 4–5 and 5+ people/m² bands | same |
| One alert threshold transfers between runs | 0.11 /s², set on runs 030/040/050, flags **90%** of pushing and 31% of walking person-seconds on runs 060/150/160, and at least one second of **75 of 78** pushing episodes (a person pushing for 2 s or more). It is set to miss little: an alert opens an investigation, never an action. Stricter thresholds, and the DM-Count pathway, are in Limitations | [results](eval/results/test_a_threshold.json), [eval/test_a_operating.py](eval/test_a_operating.py) |
| Density from video matches the reference trajectories | Pearson **0.996**, mean error 0.13 people/m² | [results](eval/results/test_a_scores.json) |
| Outflow is measured from video | **56 of 56** people counted across the entrance gate line by the console's pipeline, error 0.04 people/s | [eval/test_e1_flux.py](eval/test_e1_flux.py) |
| The two counters agree, or the system says so | DM-Count / head count inside the zone stays within **0.22 to 0.79** on 95% of 398 frames (r = 0.91); outside that band the cross-check reports disagreement | [eval/test_e1_count_band.py](eval/test_e1_count_band.py) |
| Minutes to danger arrive early and are accurate | warning **3.6 to 4.1 minutes** before the zone reaches 4/m² (5 runs); median error **32 s** when 1 to 5 minutes ahead | [eval/test_b_twin.py](eval/test_b_twin.py) |
| The forecast carries to venues it was never tuned on | 7 more street layouts (18 to 45 m long, 5 to 10 m wide, exits 1.6 to 4.0 m), 3 runs each, nothing retuned: **18 of 18** runs that reached 4/m² were warned **3.2 to 5.2 minutes** ahead, median error **35 s**; the other 3 peaked at 3.94 to 3.98/m² after a warning. Ranking actions on the zone's peak plus the error measured on the first venue found the best action for the zone in **21 of 21** runs (regret at most **0.05 people/m²**; 1.98 without the error). The console's own choice also weighs the waiting area (below) | [eval/test_b_venues.py](eval/test_b_venues.py), [results](eval/results/test_b_venues.json) |
| A surge with no trend to read is still caught | the street quiet, then arrivals jump to the full rate with no ramp, 10 layouts × 3 seeds (`eval/test_b_surge.py`): a sustained jump reached 4.5/m² in 23 evenings; all 23 had WARN and ACT **24 to 37 s** after the jump began, ACT **156 to 374 s** before 4.5 (median 196 s, against 254 s when arrivals ramp up). A 120 s pulse (400 to 1,000 people) raised ACT 24 to 41 s after it began in 30 of 30 and peaked at 4.36/m² at most | [eval/test_b_surge.py](eval/test_b_surge.py), [results](eval/results/test_b_surge.json) |
| The warnings hold when the crowd follows another pedestrian model | the same 10 layouts × 2 seeds with JuPedSim's anticipation velocity model instead of the collision-free speed model the forecaster was set on (`eval/test_b_models.py`): **19 of 19** evenings that reached 4.5/m² had WARN and ACT before it, ACT **134 to 246 s** ahead (median 181 s; 254 s on the first model). The door's forecast missed by more than the error priced from the first model (up to 3.5/m² against 2.0): in 4 of 20 evenings the zone reached 3.6 to 4.0/m² with the door open, above the 3.5 the rule aims for. It was still the least dense choice anywhere in 20 of 20: metering would have packed 4.3 to 9.5/m² into the waiting areas | [eval/test_b_models.py](eval/test_b_models.py), [results](eval/results/test_b_models.json) |
| Pricing each action's measured error ranks the zone's peaks correctly | mean regret **0.01 people/m²** on the zone's peak (out of sample), against 1.85 without | [eval/score_test_b.py](eval/score_test_b.py) |
| The proposal protects the people outside the zone too | the console weighs each action on the zone **and** on the waiting area before the gates it restricts, both priced with the largest error measured in testing. Over 30 simulated evenings on 10 layouts it opened the second door in **30 of 30**: zone peak at most **3.01/m²**, a median of **10** people held outside, and the lowest density anywhere in all 30 (the best action in hindsight). Ranked on the zone alone, it metered both gates: a median of **912** people held, up to 9.7/m² in the waiting area assumed (20 m deep across each gate) | [eval/test_b_policy.py](eval/test_b_policy.py), [src/breathingroom/choice.py](src/breathingroom/choice.py) |
| A knocked, zoomed or tilted camera is caught before its numbers are used | 216 checks on 12 moments of the recording, with rolls, zooms and tilts of known size. Two stages: an ECC screen (`findTransformECC`, homography model, on the still background) measures the floor shift to 4 mm (median) in **228 ms** on Graviton; ALIKED + LightGlue confirm in 3,046 ms, only when the screen sees motion. Together: **84 of 84** moves above 0.25 m flagged, **0** false alarms in 132 checks; flagged within **5 s** on the live instance | [eval/test_e_drift.py](eval/test_e_drift.py) |
| Counting holds on real oblique street CCTV | DM-Count QNRF on the OpenCV 5 DNN engine: **10.9%** mean error, r = 0.985 on 100 CroHD frames with 45 to 250 people | [eval/test_d_counting.py](eval/test_d_counting.py) |
| The agent makes the right kind of decision, under guards in code | **20 of 20** episodes with gpt-oss-120b on Amazon Bedrock, every proposal one the venue-wide rule accepts: a proposal for a filling zone (6/6, all "open the second exit door"), a recalibration request for a knocked camera (6/6, median 6 s), a proposal for pushing at the entrance (6/6, all the announcement "please stop pushing, the gate is open"), the deterministic planner when the model endpoint is down (2/2). Radio messages 16 to 23 words. The guards fired 3 times: twice a radio message left out the announcement it proposed and was sent back, once the model called a tool that does not exist; in every episode code forecast the catalogue actions the model had not | [eval/test_c_agent.py](eval/test_c_agent.py), [results](eval/results/test_c_agent.json) |
| Vision decides where the action goes | in all 6 pushing episodes the agent located the pushing (`locate_pushing`: a 4.6 m² region reaching 5.6/m², 1.4 m before the gate) and the radio message says where; the steward's card shows the camera crop | same |
| A wrong or manipulated model cannot put a rule-breaking proposal in front of the steward | a hostile planner tried **77** ways on the deployed tools (Zone C, Entrance E1 with pushing, the knocked camera): invented or no-action forecasts, skipped checks, actions the rule refuses, actions or metering rates the venue does not list, other zones and cameras, invented or mislabelled numbers, radio messages that ask for something else, approving its own proposal. **0** reached the steward; the rule-following planner did in both crowd scenarios and asked for recalibration on the knocked camera. Without the check that the radio message asks for the proposed action only, **14** of the 77 got through (the door's forecast, read out as "Evacuate the street now" or "Hold every gate"). Given a planted instruction to tell the stewards to evacuate and hold every gate, gpt-oss-120b tried it in **3 of 12** episodes; each was refused, and all 12 ended with a proposal the rule accepts | [eval/test_c_redteam.py](eval/test_c_redteam.py), [results](eval/results/test_c_redteam.json) |
| It runs on a small Arm instance | on a t4g.small (AWS Graviton, 2 vCPU) the console's OpenCV 5 pipeline needs **528 ms per camera-second** at half size (718 ms at full size), 873 ms with DM-Count every 5 s | [bench/bench_stages.py](bench/bench_stages.py), [results](bench/results/graviton2_t4g_small.json) |
| The DNN engine is chosen per platform | DM-Count on the zone crop takes **1,728 ms** on Graviton with `ENGINE_CLASSIC` against 9,482 ms with `ENGINE_NEW`, same count; on an x86 desktop (Zen 5) at the same 2 threads the gap is small either way (zone crop 2,057 against 1,949 ms; 1280 × 960 frame 8.4 against 6.5 s), so the slowdown is specific to Arm and the console runs the classic engine there | same, [x86](bench/results/zen5_x86_2threads.json) |
| Exit capacity comes from real bottleneck data | flow = **2.02 × width** (R² 0.81) over 24 Jülich runs; the system plans with 1.3 people/m/s, below all 24 | [eval/test_b_capacity.py](eval/test_b_capacity.py) |

![Seven venue layouts the forecast was never tuned on](docs/figures/fig_venues.png)

*Seven street layouts the forecaster was never tuned on, three runs each. Left: every run that reached 4 people/m² was warned 3.2 to 5.2 minutes ahead (the three short, wide-exit runs peaked just under 4). Right: regret on the zone's peak when actions are ranked on it, priced with the error measured on the first venue (blue) and on the forecast alone (grey); the console's venue-wide rule is scored in eval/test_b_policy.py.*

## How it works

![Architecture](docs/figures/architecture.png)

*One Graviton instance runs the OpenCV 5 pipeline, the zone physics, the tier policy and the
console. The agent calls gpt-oss-120b on Amazon Bedrock; every tier change, tool call, decision
and verification goes to a hash-chained trace mirrored to CloudWatch Logs.*

**Perception (OpenCV 5).** Each second of video gives five field samples. A frame is reduced
once with `resize` (area averaging) and undistorted with `remap` at half size, heads are found
(`cvtColor`, `inRange`, `connectedComponentsWithStats`), DIS optical flow gives their motion, and the head-plane
homography turns pixels into metres and pixels per frame into metres per second. Helbing's
crowd fields (density, velocity variance, pressure = density × variance) are three `bincount`s
and three `GaussianBlur`s on a 0.1 m floor grid. People leaving are counted across a gate line by
matching heads between samples.

**Trust.** Every 3 s, an ECC screen aligns the live frame to the calibration frame
(`findTransformECC`, coarse to fine, on a mask of the pixels that stay still in the recording) and
converts the motion of the calibration points into metres on the floor. A shift beyond the
0.25 m tolerance withdraws trust at once. Smaller motion, a loss of correlation, and every return
to trust go to the OpenCV 5 features module: ALIKED keypoints, LightGlue matching and a USAC
MAGSAC homography. The expensive check runs only when it can change the answer, and every 10
minutes regardless. On the live instance a knock is flagged within 5 s. A second, independent count comes from DM-Count on the
OpenCV 5 DNN engine, on the zone's bounding box only.

**Where.** `locate_pushing` thresholds the pushing index on the floor grid, splits it into regions
with `connectedComponentsWithStats`, places each region relative to the gate in metres, and cuts
it out of the camera image with the pushing heat drawn on it. That crop is what the steward sees
on the proposal.

**Minutes to danger.** The continuity equation over a zone, dN/dt = Q_in − Q_out, with N and
Q_out measured, gives the inflow and its trend without a camera on the entrances. Integrated
forward against the exits' capacity, it gives the minutes until 4.5 people/m² and the predicted
effect of each action, including the people an action holds back before an entrance.

**Choosing an action without moving the danger.** Metering an entrance can keep the zone clear by
filling the pavement before the gate, where no camera looks. `choice.py` judges every action on
two places: the zone (peak plus the largest error that action's forecast made in testing) and the
waiting area before the entrance it restricts (people held, times the largest shortfall of the
queue forecast in testing, over the area the venue declares in its catalogue). An action is safe
when both stay under 3.5/m². Among safe actions it takes the one that holds the fewest people
outside, then the gentlest; if none is safe, the one whose densest place is lowest. An action that
holds people where the venue has declared no waiting area is never safe.

**Decision.** A deterministic policy sets the tier from measured numbers only. At WARN, or when
the watchdog loses trust in a camera, an agent episode starts.

![Agent workflow](docs/figures/agent_loop.png)

*The model chooses which tools to call and which action to propose, and writes the radio message.
Code, not the prompt, checks every proposal: the zone's state read, its camera checked and trusted,
its count cross-checked and, when flagged for pushing, the pushing located, all in this episode; a
forecast of this episode, for this zone, of a catalogue action; every catalogue action forecast
(code forecasts any the model skipped) and the proposal one the venue-wide rule accepts; a radio
message of at most 30 words that names the zone, says pushing when there is pushing, whose
numbers each match a tool result of the same kind (a density a density, minutes to 4.5 the zone's
own), and that asks for the proposed action and nothing else, since the steward reads it out.
Metering rates and door widths are the venue's, not the model's. A filling zone cannot be closed
without action. A refusal goes back to the model as a tool
result. If the model is unreachable, a deterministic planner calls the same tools under the same
rule.*

**After the evening.** `/report` renders the record for the safety officer: every tier change,
investigation, proposal with the actions it compared, decision and who made it, and verification,
with the result of re-hashing the chain and the hash of the last entry, ready to print.

**For other agents.** `/mcp` serves the same read-only tools over the Model Context Protocol
(Streamable HTTP): zone state, the ECC camera screen, pushing hotspots, exit flow, forecasts of
catalogue actions and the latest proposals. A control room's own assistant can read the venue
with them; proposing and approving stay in the console.

## Run it

Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt          # Linux/macOS: .venv/bin/pip
.venv/Scripts/pip install -e . --no-deps
python scripts/fetch_models.py                             # ALIKED, LightGlue, DM-Count (SHA-256 checked)
python scripts/fetch_data.py --videos demo                 # Jülich data to ../data (CC BY 4.0)
BR_MODE=demo BR_USE_MODEL=0 .venv/Scripts/python -m uvicorn breathingroom.api:app --app-dir src --port 8000
```

Open <http://127.0.0.1:8000>. `BR_MODE=demo` turns on the demonstration controls and lets anyone
look, and decide while no steward accounts exist (the trace records those decisions as
unauthenticated). Without it the console runs in production mode: camera frames, evidence crops,
zone state, the record and the report are shown only to a steward signed in with their PIN, every
decision asks for the PIN again, and `/mcp` needs a bearer token.
`BR_USE_MODEL=0` uses the deterministic planner; with
`BR_USE_MODEL=1` and AWS credentials that can call the Bedrock API endpoint, the agent runs on
`openai.gpt-oss-120b`. DM-Count ONNX files come from `python scripts/fetch_models.py --export`
(needs PyTorch).

| Variable | Default | Meaning |
|---|---|---|
| `BR_DATA` | `../data` | Jülich data folder |
| `BR_USE_MODEL` | `1` | `1` agent on Amazon Bedrock, `0` deterministic planner |
| `BR_MODEL_ID` | `openai.gpt-oss-120b` | model on the Bedrock API endpoint |
| `BR_TRACE` | `var/trace.jsonl` | hash-chained trace |
| `BR_CLOUDWATCH_GROUP` | unset | also send each trace entry to this log group |
| `BR_IDLE_PAUSE_S` | `0` | pause when no console has polled for this long |
| `BR_MODE` | `production` | `production`: everything except the sign-in page and `/health` needs a signed-in steward, decisions need the PIN again, no demonstration controls; `demo`: the public demonstration |
| `BR_STEWARDS_FILE` | unset | steward accounts, JSON name → PIN hash, made with `python -m breathingroom.stewards "<name>"`; each decision is recorded under the steward whose PIN was entered |
| `BR_STEWARD_PIN` | unset | a single account named `steward` |
| `BR_SESSION_SECRET`, `BR_SESSION_HOURS` | random, `12` | key that signs sign-in cookies (random: a restart signs everyone out); how long a sign-in lasts |
| `BR_MCP_TOKEN` | unset | bearer token for `/mcp` in production; without it the endpoint is closed there |
| `BR_EVIDENCE_TTL_S` | `900` | how long a camera crop of a pushing spot is served (crops are kept in memory only, the 24 newest) |
| `BR_FEED_STALE_S` | `10` | a zone whose source has produced nothing for this long is marked lost: untrusted on the console, not investigated, and `/health` answers 503 |
| `BR_TRACE_KEY` | unset | Ed25519 private key (PEM) that signs every trace entry; the public key is at `/api/trace/key` |
| `BR_ISOLATE_MODELS` | `0` | `1` runs ALIKED + LightGlue and DM-Count in a fresh child process (small instances) |
| `BR_SCREEN_EVERY_S`, `BR_CONFIRM_EVERY_S` | `3`, `600` | ECC screen period; features check period while the screen is steady |
| `BR_WORK_SCALE` | `0.5` | fraction of full size at which camera frames are undistorted and searched |

## Test it

```bash
.venv/Scripts/python -m pytest -q                         # 36 unit tests
python tools/ui_smoke.py http://127.0.0.1:8000            # drives the console headless: knock, recalibration, proposal, approve, verify
```

The evaluations write to `eval/results/`:

```bash
python eval/test_a_pushing.py && python eval/score_test_a.py   # pushing index vs ratings (6 runs)
python eval/test_a_console.py                                  # the same, through the console's CameraZone
python eval/test_b_capacity.py                                 # exit capacity, 24 bottleneck runs
python eval/test_b_twin.py && python eval/score_test_b.py      # forecasts and actions on the venue twin
python eval/test_b_venues.py                                   # the same on 7 other layouts
python eval/test_b_policy.py                                   # the console's alert policy, and the queue each action creates
python eval/test_b_surge.py                                    # sudden surges with no trend to read
python eval/test_b_models.py                                   # the same evenings with a second pedestrian model
python eval/test_a_operating.py                                # index plus density; alert operating points
python eval/test_c_agent.py --repeats 6                        # agent episodes on fixed scenarios
python eval/test_c_redteam.py --model 6                        # a hostile planner against the guards; the model with a planted instruction
python eval/test_d_counting.py                                 # DM-Count on CroHD (download HT21.zip to ../data/ht21)
python eval/test_e_drift.py                                    # camera watchdog against knocks of known size
python eval/test_e1_flux.py                                    # gate-line outflow against trajectories
python eval/test_e1_count_band.py                              # agreement band of the two counters
python eval/figures.py                                         # figures in docs/figures
```

## Deploy on AWS

```bash
AWS_PROFILE=<admin> bash deploy/provision.sh                    # bucket, log group, role, security group, instance, Elastic IP
AWS_PROFILE=<admin> bash deploy/redeploy.sh <instance-id> <a-b-c-d>.sslip.io
```

`provision.sh` creates an IAM role limited to Systems Manager, the Bedrock API endpoint,
one log group (append and read, no delete) and one bucket; a security group open on 80 and 443 only (no SSH: the instance is
administered through SSM); and a t4g.small (Graviton) on Amazon Linux 2023 with IMDSv2 required
and an encrypted volume. `setup_instance.sh` installs the pinned dependencies with the headless
OpenCV 5.0.0 aarch64 wheel, builds JuPedSim from source (PyPI has no aarch64 Linux wheel), and
runs the console under systemd behind Caddy, which obtains a Let's Encrypt certificate for the
sslip.io hostname. On first setup it makes the Ed25519 key that signs the trace, in
`/etc/breathingroom`, readable but not writable by the service. The public instance runs with
`BR_MODE=demo` so visitors can try the approval; a venue would run in production mode with
named steward accounts.

## Repository

| Path | What it holds |
|---|---|
| `src/breathingroom/calib.py` | floor homography (USAC MAGSAC), focal length and pose from the floor, head-plane homography |
| `src/breathingroom/fields.py` | Helbing density, velocity, variance and pressure fields on a metric grid |
| `src/breathingroom/flow.py`, `density.py`, `drift.py` | DIS optical flow; DM-Count on the DNN engine; ECC screen and ALIKED + LightGlue watchdog |
| `src/breathingroom/physics.py` | zone mass balance, forecast, actions |
| `src/breathingroom/policy.py`, `agent.py` | deterministic tiers; agent episode, tools, guards, fallback planner |
| `src/breathingroom/runtime.py`, `twin.py`, `api.py`, `web/` | live zones, venue twin (JuPedSim), API and console |
| `src/breathingroom/trace.py`, `report.py` | hash-chained, signed JSONL trace, checked against its CloudWatch Logs copy; the evening report |
| `src/breathingroom/stewards.py` | steward accounts: PIN hashes, wrong-PIN lockout |
| `src/breathingroom/mcp.py` | the read-only tools over the Model Context Protocol |
| `eval/`, `bench/`, `tests/`, `tools/` | evaluations, per-stage timing, unit tests, UI driver |
| `deploy/` | AWS provisioning, instance setup, systemd and Caddy units |

## Data and models

- Forschungszentrum Jülich pedestrian data, CC BY 4.0: 2018 entrance experiments (DOI
  10.34735/ped.2018.1), pushing ratings (DOI 10.34735/ped.2022.4), 2021 bottleneck experiments
  (DOI 10.34735/ped.2021.9). The console replays run 150 of the entrance experiments.
- The lens model and per-run homographies were recovered from the videos themselves
  (`eval/fzj_calibrate.py`): lens from SIFT matches against Alia et al.'s undistorted video,
  homography from their published pixel tracks, refined per run by matching caps to trajectories.
  Runs 030, 040, 050, 060, 150 and 160 calibrate at 2.3 to 5.6 px median error.
- CroHD / HT21 (CC BY-NC-SA 3.0) for test D only; not redistributed.
- DM-Count (MIT), ALIKED and LightGlue ONNX as pinned by opencv_extra.

## Limitations

- **Pushing detection is shown on overhead video of one experiment series, not on arbitrary
  CCTV.** The E1 head detector finds the orange caps every participant wore in the Jülich
  experiments. On ordinary CCTV, heads would come from a counting network: DM-Count QNRF measured
  10.9% error on oblique street footage, but no rated pushing data from an oblique camera was
  available, so pushing on such views is untested. On the overhead view, held out
  (`eval/test_a_operating.py`, 78 pushing episodes):

  | Pathway, threshold | Pushing person-seconds | Walking person-seconds | Pushing episodes |
  |---|---|---|---|
  | heads, set for 10% walking | 36% | 5% | 63% |
  | heads, set for 20% walking | 63% | 13% | 81% |
  | heads, the console's 0.11 /s² | 90% | 29% | 96% |
  | DM-Count, set for 10% walking | 12% | 3% | 44% |
  | DM-Count, set for 20% walking | 30% | 8% | 65% |
  | DM-Count, its Youden point | 72% | 25% | 85% |

  The console runs the sensitive point: an alert opens an investigation and a person looks at the
  crop; nothing acts on its own. A threshold strict enough for unattended alerts misses most
  pushing, and an approval step cannot make up for pushing the camera never flags. With the head
  detector, the index and density together reach AUROC 0.831 against 0.786 for density alone.
- The ECC screen models a camera that rotates and shifts. A zoom or a tilt shows up as a drop in
  correlation, which sends the frame to the features check.
- The venue twin's pedestrian model cannot pack people above about 4.5/m²: queues spill upstream,
  so forecast peaks above that are not scored. Forecasts of partial actions (one gate, one extra
  door) were optimistic by about 2 people/m² in the twin. The proposal ranks actions on predicted
  peak plus that measured error, and the verification step catches an action that does not bend
  the curve.
- The pushing threshold was set and tested on six runs of one experiment series.
- Forecasts and actions are scored in simulation (JuPedSim), on scenarios from the same simulator:
  consistent within the model, not shown at real events. Under the console's own alert policy
  (WARN at 10 minutes to 4.5/m², ACT at 5), 30 simulated evenings: all 22 that reached 4.5/m²
  had WARN 247 to 457 s ahead; the other 8 peaked at 3.94 to 4.48/m² after an ACT
  (`eval/test_b_policy.py`).
- Forecast errors were priced on one pedestrian model. On a second one the door's error was
  larger than priced, and in 4 of 20 evenings the zone reached 3.6 to 4.0/m² after the rule's
  choice (`eval/test_b_models.py`); every simulator caps out near 4.5 to 6/m², below the
  densities of a crush, and no forecast here has met a real crowd.
- The venue-wide rule weighs the waiting area before each restricted entrance, but that area is a
  number the venue declares (the demonstration assumes 60 m² per gate), not something a camera
  measures; approach routes, alternative entrances and the consequences of closing one are not
  modelled. Its forecast errors and queue factors were measured on one simulated venue and carried
  to the other nine. Ranked on the zone's peak alone, the same evenings would have held a median
  of 912 people outside.
- The camera zone replays a recording: approving an action there records the decision only.

## Responsible use

**What is kept, and what is shown.** The measurements and the record hold aggregate numbers only:
density, the pushing index, counts and flows on a floor grid. No faces, no identities, no
tracking of a person, and no video or image is written to disk or to the record. Images are
shown, though: the console displays the latest camera frame with the pushing field drawn on it,
and each pushing proposal carries a crop of the spot so the steward can see what the index saw.
Both are images of people. The frame is held in memory until the next one replaces it; only the
24 newest crops are held, in memory, each served for 15 minutes at most and with `no-store`. On the public
demonstration they show the Jülich recording and anyone can view them; in production
(`BR_MODE=production`) every image, the zone state, the record and the report need a signed-in
steward, and the MCP endpoint its own token.

**Who decides.** The same measurements could be used to disperse a lawful gathering, so the system
is meant to be operated by venue safety staff under a written purpose. Nothing reaches a gate, a
door or a public-address system without a person's approval, and every recommendation, decision
and outcome stays in a tamper-evident trace.

**What a venue would still need.** The deployment is one t4g.small: it shows the pipeline runs on
a small Arm instance, not that it serves a venue's cameras without interruption. A zone whose
source stops is marked lost on the console and `/health` answers 503, for a monitor outside the
instance to raise an alarm; there is no second instance, no failover, no sustained-load test with
many cameras and several concurrent episodes, and no retention policy beyond the 30 days of the
log group. Those are part of a pilot, as is a data-protection assessment for the venue.

## Licence

Apache-2.0.
