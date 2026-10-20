# Breathing Room: technical report

Marc Donovici · OpenCV AI Competition 2026 · October 2026

Breathing Room measures three things per zone from camera video: density in people per m², a
pushing index from crowd motion, and minutes until the zone reaches 4.5 people per m² at the
current inflow. An agent turns a warning into one proposed action with its predicted effect and
a camera crop of where the pushing is, a steward approves or rejects it, and the system checks
90 seconds later whether the action worked. The record of the evening is hash-chained and
rendered as a report for the safety officer. This report covers the problem, the users, the architecture, the OpenCV 5
implementation, the AWS deployment, the evaluation with its failures, the limitations and
responsible use.

## 1. Problem

Crowd crushes build up over minutes, and they are visible on camera before anyone dies. At
Itaewon, Seoul, on 29 October 2022, the first of 11 emergency calls warning of a crush came at
18:34. Police investigators later estimated from security footage about 8 people per m² in the
alley at 22:15 and 9 to 11 per m² by 22:25; 159 people died. Since 2023 Seoul has run software
that counts people on CCTV and alerts the district safety room.

A head count answers one question: how full is this street? Stewards need two more answers:

1. **Has the crowd started to push?** Pushing turns a dense crowd into a dangerous one. Two
   crowds at 4 people per m² can be in very different states.
2. **How long until it becomes dangerous, and what would change that?** A density reading with
   no time and no options leaves the steward guessing.

Prior work at Forschungszentrum Jülich detects pushing at event entrances from video with a
trained classifier (Alia et al., IEEE Access 2023, 87% accuracy on their test set). Breathing
Room takes a different route: a physical index that needs no training, measured in metric units,
combined with a forecast and a closed loop with a person in charge.

## 2. Users

- **Stewards and crowd managers at events, stations and venues.** They need a short message they
  can read over the radio, a single action to approve, and confirmation that it worked.
- **Safety officers who review an evening afterwards.** They need a record of what the system
  measured, what it proposed, who decided, and what happened, that nobody can quietly edit.

## 3. Architecture

![Architecture](figures/architecture.png)

*The OpenCV 5 pipeline, zone physics, tier policy, agent loop and console run on one Graviton
instance; the agent's model is gpt-oss-120b on Amazon Bedrock; the trace is mirrored to
CloudWatch Logs.*

Two zones run side by side in the console:

- **Entrance E1** replays a real overhead recording (Jülich entrance experiment, run 150) through
  the full vision pipeline. Actions there are recorded but cannot change the footage.
- **Zone C, Station Road** is a venue twin simulated with JuPedSim: a 30 × 6 m street fed by two
  gates and ending in a 2.4 m exit, with a second 1.5 m exit door that can be opened. Approved
  actions change the simulated crowd, so the verification step has something to measure.

Each zone runs on its own thread. Heavy work happens outside the runtime lock and the state is
committed under it, so the console always reads a consistent snapshot.

## 4. OpenCV 5 implementation

| Step | OpenCV 5 | Notes |
|---|---|---|
| Decode | `VideoCapture` | read forwards with `grab`/`read`: seeking per frame cost 75 ms and could abort the FFmpeg decoder |
| Working size | `resize` (`INTER_AREA`) | each frame is reduced to half size once; the camera matrices are scaled to match, so undistortion and head search run on a quarter of the pixels |
| Lens | `initUndistortRectifyMap`, `remap`, `undistortPoints`, `projectPoints` | lens recovered from SIFT matches between the original and a published undistorted video (0.29 px median) |
| Heads | `cvtColor` (HSV), `inRange`, `connectedComponentsWithStats` | the Jülich participants wore orange caps; heads are returned in full-size undistorted pixels |
| Motion | `DISOpticalFlow` (medium, 960 px) | flow between frames 0.08 s apart |
| Metric geometry | `findHomography` (USAC MAGSAC, RANSAC fallback), `perspectiveTransform` | head-plane homography; focal length and camera pose recovered from the floor homography for oblique views |
| Crowd fields | `GaussianBlur` on a 0.1 m grid | Helbing et al. (2007): density, velocity variance, pressure = density × variance |
| Outflow | Hungarian matching of heads between samples (SciPy), line test in metres | heads crossing a line 0.4 m before the gate |
| Second count | `dnn.readNetFromONNX` with the engine picked per architecture (`ENGINE_CLASSIC` on Graviton, `ENGINE_NEW` on x86), `blobFromImage` | DM-Count (SHA for the overhead view, QNRF for oblique street CCTV) |
| Camera screen | `findTransformECC` (`MOTION_HOMOGRAPHY`, coarse to fine, with a mask) | every 3 s; the mask keeps the pixels that stay still in the recording (barriers, floor, walls) |
| Camera confirmation | `ALIKED_create`, `LightGlueMatcher_create` (features module), `findHomography` (USAC MAGSAC) | when the screen sees motion, and every 10 minutes; both stages move the calibration points and measure their shift on the floor in metres |
| Where the pushing is | `connectedComponentsWithStats` on the thresholded field, `warpPerspective`, `imencode` | regions placed relative to the gate in metres, and cut out of the camera image for the steward |
| Overlay | `warpPerspective`, `applyColorMap`, `polylines`, `imencode` | the pushing field is drawn back onto the image through the homography |

**Pushing index.** For each person, Helbing's density ρ and the velocity variance within a
Gaussian kernel of R = 0.7 m are averaged over a 2 s window; the pressure is ρ × Var(v) in 1/s².
On a grid of cell area a, the kernel sum is a Gaussian blur of the per-cell people mass divided
by a, so the whole computation is three `bincount`s and three `GaussianBlur`s per frame. The
alert threshold, 0.11 /s², was set on three runs and tested on three others (section 6.1).

**Minutes to danger.** The continuity equation over a zone, dN/dt = Q_in − Q_out, uses two
measured quantities (the people in the zone, the flow leaving it) and gives the inflow and its
trend without a camera on the entrances. The forecast carries the inflow trend forward for two
minutes (to at most 1.5 times its current value), lets the outflow scale with occupancy up to
the exits' capacity (1.3 people per metre per second, below every one of 24 measured bottleneck
runs), and integrates in 5 s steps. An action changes one term: metering a gate caps its share of
the inflow, holding it removes that share, opening a door adds capacity.

## 5. The agent

![Agent workflow](figures/agent_loop.png)

*Perception, a deterministic tier, an investigation with OpenCV tools, and a person who decides.
Dashed red paths are the failure branches.*

- **Who decides what.** The tier comes from deterministic rules on measured numbers. The model
  (gpt-oss-120b on the Amazon Bedrock API endpoint) decides which tools to call, which action to
  propose, and writes the radio message. A steward approves or rejects. Nothing reaches a gate or
  a public-address system without that.
- **Tools.** `get_zone_state`, `check_camera` (ECC screen, confirmed by ALIKED + LightGlue when
  it sees motion), `cross_check_count` (DM-Count against the head count, with an agreement band
  measured on six runs), `locate_pushing` (hotspots in metres from the gate, with a camera crop),
  `measure_exit_flow`, `forecast_action`, `propose_action`, `request_recalibration`, `no_action`. Every forecast in one episode uses the same frozen
  history, so the compared actions are comparable.
- **Vision changes what happens next.** A camera that fails the watchdog stops counting: the
  agent asks for recalibration and proposes no crowd action from it. A count that disagrees with
  the second counter is reported as such. `locate_pushing` decides where the action goes: the
  radio message names the side and the distance before the gate, and the steward sees the crop.
  The forecast drives which action is proposed.
- **Guards in code, not in the prompt.** `Episode._propose` refuses a proposal unless, in this
  episode: the zone's state was read and its feed is live; its camera was checked and was trusted
  at the latest check; its count was cross-checked; and, when the zone is flagged for pushing,
  `locate_pushing` ran. Every tool call is held to the episode's zone and camera; the cited
  forecast must be of this episode, for this zone, of an action in the catalogue. Before judging
  the proposal, code forecasts every catalogue action the model did not (marked "by the guard" in
  the trace and on the console), so a gentler or safer action cannot be missed because it was
  never compared; the proposed action must then be one the venue-wide rule accepts (below). The
  radio message has at most 30 words, names the zone, says pushing is detected when the zone is
  flagged for it, and each number must match a tool result of the same kind, read from the words
  after it: a density only a density field, "minutes to/left/until" only the zone's own
  `minutes_to_4_5`, a rate a rate, metres a distance, people a count. Reusing a true number as
  another quantity (a count as a density, a density as minutes) is refused; a number with no unit
  only has to match some tool result. A zone whose own state says it reaches 4.5/m² within 10
  minutes cannot be closed without action. A refusal goes back to the model as a tool result, and
  the model corrects itself.
- **The venue-wide rule (`choice.py`).** Each catalogue action carries the mean and the largest
  error its forecast made in test B, and, for actions that hold people back, the largest factor
  by which the queue forecast fell short there, and the waiting area the venue declared before
  that entrance. An action is safe when the zone's peak plus the largest error, and the queue
  priced the same way over its waiting area, both stay under 3.5/m² (WARN). Among safe actions the
  rule takes the one that holds the fewest people outside, then the least disruptive, then the
  lowest zone peak; if none is safe, the one whose densest place, zone or waiting area, is lowest,
  the gentlest within 0.25/m². An action that holds people back where no waiting area is declared
  is never safe. The deterministic planner, the guard and the evaluation (6.4b) use the same code.
  Ranking on the zone's peak alone would keep the zone clear by holding hundreds of people before
  the gates (6.4b).
- **Failure handling.** A failing tool is reported to the model, never raised. An unreachable
  model hands the episode to a deterministic planner that calls the same tools. A proposal nobody
  decides on expires after 4 minutes. An approved action that does not bend the measured curve
  within 90 s opens a new episode at ACT.
- **Observability.** Every tier change, tool call (arguments, result, milliseconds), proposal,
  decision and verification is appended to a JSONL trace in which each entry carries the SHA-256
  of the previous one; `/api/trace/verify` re-hashes the chain. The same entries go to CloudWatch
  Logs.

Example from the trace: the model asked to forecast `meter_entry` with no target at Entrance E1;
the guard answered with the list of available actions, and the next call named "queue feed"
correctly.

## 6. Evaluation

All results are produced by scripts in `eval/` and stored in `eval/results/`.

### 6.1 Test A: the pushing index against trained raters

Data: six calibrated runs of the Jülich 2018 entrance experiments with per-person, per-second
pushing ratings by trained raters (DOI 10.34735/ped.2022.4): 9,260 rated person-seconds, 342
people, 34% pushing. The index is computed from video alone (undistortion, heads, DIS flow,
homography, fields) and sampled at each rated person's position. Nothing is trained.

| Signal | AUROC |
|---|---|
| Pushing index from video (heads) | **0.798** [0.767, 0.828] |
| Density from video | 0.786 |
| Pushing index from the reference trajectories | 0.776 |
| Pushing index from DM-Count density | 0.767 |
| Pushing index from the console's `CameraZone`, at half resolution | 0.797 |

![Test A by density band](figures/fig_test_a_bands.png)

*Within each density band, where a head count carries no information, the pushing index still
separates pushing from walking.*

`eval/test_a_pushing.py` builds the fields with its own loop. `eval/test_a_console.py` runs the
console's `CameraZone.tick` itself, at the half resolution the console runs at, over the same six
runs and 9,260 rows: AUROC 0.797, and 0.87, 0.85, 0.74, 0.73 and 0.69 within the density bands.
At full resolution the same code gives 0.798. The deployed pipeline is the evaluated one.

The threshold 0.11 /s² is the Youden point on runs 030, 040 and 050. On runs 060, 150 and 160 it
flags 90% of pushing person-seconds and 31% of walking ones. Video density against the reference
trajectories: Pearson 0.996, mean absolute error 0.13 people/m².

### 6.1b Does the index add to density, and how specific can an alert be?

`eval/test_a_operating.py`, on the same 9,260 rows.

- **Together, not instead.** Alone, the index (0.798) is 0.012 above video density (0.786).
  Together, in a logistic model fitted on five runs and scored on the sixth (leave one run out),
  they reach **0.831**: 0.045 above density alone, 95% interval 0.031 to 0.060 (resampling
  people). The index carries motion information that density does not. DM-Count reads density
  2.4/m² low on this overhead view (r = 0.75), and there the pair gains nothing over the index
  alone (0.709 against 0.767).
- **Operating points.** The threshold is set on runs 030, 040 and 050 to flag a target share of
  walking person-seconds, with the index averaged over each person's last 3 s, and scored on runs
  060, 150 and 160:

| Pathway | Set to flag, walking | Held-out pushing flagged | Held-out walking flagged |
|---|---|---|---|
| heads | 10% | 36% | 5% |
| heads | 20% | 63% | 13% |
| DM-Count | 10% | 12% | 3% |
| DM-Count | 20% | 30% | 8% |

Person-seconds understate what a steward needs, which is to learn that a stretch of pushing is
happening. Counting a person's run of 2 or more seconds rated pushing as one episode (78 on the
held-out runs), and an episode as detected when any of its seconds is flagged:

| Pathway | Threshold | Pushing person-seconds | Walking person-seconds | Pushing episodes detected |
|---|---|---|---|---|
| heads | set for 10% walking | 36% | 5% | 63% |
| heads | set for 20% walking | 63% | 13% | 81% |
| heads | **the console's, 0.11 /s²** | **90%** | 29% | **96% (75 of 78)** |
| DM-Count | set for 10% walking | 12% | 3% | 44% |
| DM-Count | set for 20% walking | 30% | 8% | 65% |
| DM-Count | its Youden point, 0.035 /s² | 72% | 25% | 85% |

(Index averaged over each person's last 3 s; the console's threshold flags 31% of walking
person-seconds without that averaging.) The console runs the sensitive point on purpose: an alert
opens an investigation that ends with a camera crop in front of a steward, and never acts by
itself, so a false alert costs a look and a missed one costs the warning. At thresholds strict
enough for unattended alerts, most pushing is missed, on the DM-Count pathway by a wide margin.
What these numbers do not show: pushing detection on ordinary oblique CCTV. Every rated pushing
second here comes from overhead video of one experiment series; DM-Count on oblique street
footage is tested for counting only (6.6).

### 6.2 Outflow from video

Heads counted across the gate line in run 150 by the console's pipeline: 56 people, the same 56
as the trajectories; mean 1.00 people/s against 1.00, error 0.04 people/s, r = 0.947 on 5 s
averages. At half resolution a morphological opening erases the smallest caps and the count
falls to 55, so the half-size detector keeps the area test and has no opening. Integrating the
fields' flux through the line instead reads 40% low, which is why heads are matched and counted.

![E1 outflow](figures/fig_e1_outflow.png)

*Outflow through the E1 gate, measured from video and from the published trajectories.*

### 6.2b Agreement of the two counters

`cross_check_count` runs DM-Count on the zone's bounding box at half resolution (710 ms per count
on the desktop, against 6.9 s for the whole frame at 1280 px) and compares it with the head count
in the zone. Over 398 frames of the six runs, every 0.8 s with at least 3 heads in the zone, the
ratio DM-Count / heads has median 0.45 and stays within 0.22 to 0.79 on 95% of frames (r = 0.91).
That band is the console's definition of agreement.

### 6.3 Exit capacity

Over 24 runs of the Jülich 2021 bottleneck experiments, steady flow = 2.02 × width (R² 0.81);
specific flow 1.41 to 2.69 people per metre per second. The forecast uses 1.3, below all 24.

### 6.4 Test B: minutes to danger and the effect of actions (venue twin)

Five evenings (seeds) of the twin. Every second the forecaster sees only the past.

- **Lead time.** The forecast first says "5 minutes to 4.0/m²" 218 to 247 s before the zone
  reaches it. Median error of the predicted time when the crossing is 1 to 5 minutes ahead: 32 s.
- **Actions.** At the warning, each run is branched once per action and simulated for 5 minutes.

| Action | Predicted peak | Simulated peak | Mean error |
|---|---|---|---|
| no action | 4.15 | 4.30 | 0.26 |
| meter both gates to 150/min | 0.58 | 0.69 | 0.11 |
| hold both gates | 0.58 | 0.68 | 0.10 |
| meter gate A to 60/min | 0.87 | 2.86 | 1.99 |
| open the second exit door | 0.61 | 2.53 | 1.92 |

![Twin actions](figures/fig_twin_actions.png)

*Seed 0: simulated density (solid) and forecast (dashed) after the warning.*

The forecast is accurate for actions on the whole inflow and optimistic for partial ones: in the
twin, each exit and each gate serves its own side of the street, and the arrivals keep rising.
Ranking actions on predicted peak alone would choose the second door (least disruptive) and
miss by 1.85 people/m² on average. Ranking on predicted peak plus each action's measured error,
estimated on the other four evenings (leave one out), chooses metering both gates in 5 of 5
evenings, with a mean regret of 0.01 people/m² on the zone's peak. That ranking looks at the zone
only; 6.4b shows what it does to the people held outside, and the rule the console uses now.

**Other venues.** The same forecaster and the same choice rule ran, with nothing retuned, on seven
more layouts (`eval/test_b_venues.py`): streets 18 to 45 m long and 5 to 10 m wide, main exits
1.6 to 4.0 m, second doors 1.0 to 2.0 m, surges scaled to the exit, three seeds each. Metering
rates scale with the exit width; each action is priced with the error measured on the first venue
only, as the console does.

- 18 of the 21 runs reached 4.0/m². All 18 were warned 3.2 to 5.2 minutes ahead (median 3.5);
  median error of the predicted time 1 to 5 minutes ahead: 35 s. The three runs of the short street
  with a 3 m exit peaked at 3.94 to 3.98/m², just under the line, after a warning.
- Ranking on predicted peak plus the first venue's error found the best action for the zone in
  21 of 21 runs: regret at most 0.05 people/m² (mean 0.01). Ranking on predicted peak alone chose
  the second door every time, with a mean regret of 1.98. Both rankings look at the zone only
  (6.4b).

![Seven more venues](figures/fig_venues.png)

*Lead time of every warning that preceded 4 people/m² (left) and regret of the chosen action with and without the first venue's measured error (right), seven layouts, three seeds each.*

Building the twin exposed two simulator problems: with the collision-free speed model, streams
that cross at an exit gridlock, and a person bound for one door who is pushed into the other
keeps pressing against the wall between them. A side passage halfway along the street makes the
crowd worse when opened, so the twin's extra exit is a second door next to the main exit, and
people near the exits use the open door they are lined up with.

### 6.4b Under the console's alert policy, and the crowd an action moves upstream

The tests above score a warning given 5 minutes before 4.0/m². The console works differently:
`TierPolicy` raises WARN, which opens an investigation, at 10 minutes to 4.5/m² (or 3.5/m²
measured) and ACT at 5 minutes to 4.5/m² (or 4.5/m² measured). `eval/test_b_policy.py` replays
the twin through `TierPolicy` itself, with the console's forecast settings, on the reference
venue, the seven layouts above, and two with a second door wider than usual (2.8 m beside a
1.6 m exit; 2.2 m beside 1.8 m): ten layouts, three seeds, 30 evenings.

- **Alerts.** 22 evenings reached 4.5/m². In all 22, WARN came 247 to 457 s ahead (median 275 s)
  and ACT 212 to 432 s ahead (median 255 s). The other 8 peaked at 3.94 to 4.48/m², after both
  WARN and ACT: the forecast said 5 minutes to 4.5 and the crowd stopped just short of it.
- **Metering moves the crowd upstream.** The twin counts the people a gate holds back. Metering
  both gates kept a median of 912 people waiting before the gates (up to 1,626), holding both
  1,559 (up to 2,876), opening the second door 10 (up to 268). The twin does not model the
  approach, so the queue's density is its count over an assumed waiting area: 20 m deep across
  each gate's half of the street (60 m² per gate on the reference venue, as the console assumes).
- **The forecast of that queue is low.** In all 120 action runs the forecast queue was at or
  below the simulated one, by a median factor of 1.6 (holding) to 2.8 (metering both gates): the
  forecast carries the inflow seen at the warning, and the surge kept rising. The console prices
  each queue with the largest factor seen on the reference venue (2.87 metering both, 2.29
  metering one, 1.58 holding).
- **Zone only against venue-wide.** A rule that ranks actions on the zone's peak alone (least
  disruptive within 0.25/m² of the lowest peak plus mean error) meters both gates in 30 of 30
  evenings: the zone stays under 0.69/m², but a median of 912 people wait outside, up to 9.7/m²
  in the waiting area, far denser than the zone it protects. The console's venue-wide rule
  (section 5) opens the second door in 30 of 30: the zone peaks at 3.01/m²
  at most, under WARN in all 30; a median of 10 people wait outside (up to 268, 2.2/m², in three
  evenings of the short street whose gates back up on their own: 499 to 523 wait there with no
  action). In all 30 its choice had the lowest density anywhere, zone or waiting area, of the
  four actions in hindsight; the zone-only rule was 5.2/m² worse on that measure on average.
- **What this does not settle.** The door works here because every layout has one. Where no
  action is safe, the rule takes the least dense place, which can still be dense; where the venue
  declares a large waiting area, metering becomes safe and may be chosen. The waiting areas, the
  queue factors and the safety bar of 3.5/m² are inputs a venue's safety plan must own, and
  approach routes, alternative entrances and the consequences of closing one are not modelled.

### 6.4c Sudden surges

The forecast carries the inflow trend of the last 45 s forward for two minutes, so its weak point
is a surge with no trend before it: a train unloading, a hall letting out. `eval/test_b_surge.py`
runs the ten layouts of 6.4b with the street quiet (1.5 arrivals per second, scaled by each
layout's demand) until 180 s, then the full rate of the test B evenings (6.5 per second, scaled)
with no ramp, through the console's `TierPolicy`. Three seeds each.

| Surge | Evenings | Reached 4.5/m² | WARN and ACT after the surge began | ACT before 4.5/m² |
|---|---|---|---|---|
| step: the full rate to the end (1,400 to 3,500 extra people) | 30 | 23 | 22 to 34 s (WARN), 24 to 37 s (ACT) | 23 of 23, 156 to 374 s ahead, median 196 s |
| pulse: the full rate for 120 s (400 to 1,000 people) | 30 | 0 (peak 2.1 to 4.4) | 22 to 38 s (WARN), 24 to 41 s (ACT), all 30 | |

On the same layouts with the arrivals ramping up (6.4b) ACT came a median 254 s ahead, so a surge
with no warning in its trend costs about a minute of lead time here. None of the 60 quiet
streets was warned before its surge. What this does not cover: a surge the zone's own camera
cannot see until it arrives, faster than these, or one that starts already above 4.5/m² at the
entrance; upstream cameras would see it sooner.

### 6.4d A second pedestrian model

Every test above scores the forecast against JuPedSim's collision-free speed model, on which the
forecaster's settings were chosen, so the forecast and its ground truth could share that model's
habits. `eval/test_b_models.py` reruns ten layouts, two seeds each, with JuPedSim's anticipation
velocity model, in which people anticipate their neighbours and crossing streams keep moving;
forecaster, policy, layouts and arrivals unchanged.

| | Collision-free speed (same evenings) | Anticipation velocity |
|---|---|---|
| Evenings reaching 4.5/m² | 14 of 20 | 19 of 20 |
| WARN and ACT before 4.5/m² | 14 of 14 | 19 of 19 |
| ACT lead, s (median) | 212 to 392 (254) | 134 to 246 (181) |
| Door: zone peak, people/m² (max) | 2.96 | 4.02 |
| Door: forecast error, people/m² (max) | 2.43 | 3.54 |
| Metering both: people held (median) | 902 | 907 |

The second model packs the street faster, so warnings come about a minute later, still before
4.5/m² in every evening. The forecast of the door was more optimistic than the error priced from
the first model (1.96 at most): in 4 of 20 evenings (the long street with a 1.6 m exit, and the
plaza) the zone reached 3.6 to 4.0/m² with the door open, above the 3.5/m² the rule aims for and
below 4.5. In all 20, the door still left the lowest density anywhere: metering would have kept
the zone under 0.9/m² by putting 4.3 to 9.5/m² into the assumed waiting areas. The console's 90 s
verification compares the measured curve with the forecast one and escalates when it does not
bend; these runs do not exercise it.

Scenarios and scoring come from the same simulator family (JuPedSim). These results show the
forecast and the choice are consistent within that model, not that they work at real events.

### 6.5 Test C: the agent

20 episodes on fixed scenarios built from the real pipeline, gpt-oss-120b on Amazon Bedrock. Each
episode starts with the reasons the tier policy gives for the zone's measured state.

| Scenario | Episodes | Expected decision | Outcome |
|---|---|---|---|
| Zone C filling (WARN, about 9 minutes to 4.5/m²) | 6 | proposal | 6/6, all "open the second exit door", each citing its own forecast and accepted by the venue-wide rule |
| Entrance E1, pushing, density not rising | 6 | proposal | 6/6: 4 announcements, 2 stewards sent; in 6/6 the agent called `locate_pushing` (a 4.6 m² region reaching 5.6/m², 1.4 m before the gate) and the radio message says where the pushing is |
| Entrance E1, camera knocked | 6 | recalibration request | 6/6, median 3 tool calls and 7 s |
| Zone C, model endpoint unreachable | 2 | deterministic planner | 2/2, the same door under the same rule |

20 of 20 overall; a crowd episode takes a
median 10 tool calls and 11 to 22 s; radio messages have 16 to 23 words. Four calls were refused:
twice the model proposed an announcement at E1 before calling `cross_check_count` and was told
so, and twice it asked to forecast an announcement in Zone C, where the catalogue has none. The
model forecast 1 to 4 actions per episode; code forecast the rest of the catalogue before
judging each proposal (1 to 4 per episode), so every proposal was compared with every action.
One message gave the hotspot's own density ("5.6 /m² at centre 1.4 m before gate"), which the
kind-aware check accepts because `locate_pushing` returned it as a density.

What each guard is for, from what a model can get wrong: closing a filling zone with
`no_action` by reading an action's forecast (0.55/m²) as the zone's future while
`get_zone_state` says 9.1 minutes to 4.5/m² (refused unless a no-action forecast shows the zone
stays below); writing "stable for 10 minutes" when no tool returned 10; taking the minutes left
from an action's forecast instead of the zone's state; proposing to hold both gates when a
gentler action is as safe; proposing before the camera and the count are confirmed. A tool that
fails inside an episode (for instance out of memory in `check_camera`) is reported to the model,
and an unverified camera leads to a recalibration request, not a proposal.

### 6.6 Test D: counting on oblique street CCTV

CroHD (HT21 training sequences): four real, dense, oblique scenes; 25 frames each, every head
annotated (45 to 251 people per frame).

| DM-Count training | Mean absolute error | Mean % error | r |
|---|---|---|---|
| QNRF | 13.4 | **10.9%** | 0.985 |
| SHA | 48.0 | 23.6% | 0.856 |
| SHB | 113.6 | 61.1% | 0.588 |

On the overhead Jülich view DM-Count undercounts (SHA counts 0.27 to 0.99 times the head count),
which is why E1 uses it as a second counter with an agreement band measured on six runs.

### 6.7 Test E: the camera watchdog

216 checks: 12 moments of run 150 against the calibration frame, each as recorded and after
moves of known size: a roll of 0.25 to 4 degrees with a proportional shift, a zoom of 1 to 8%,
and a tilt that narrows one edge of the image by 1 to 8%. The true floor shift of the
calibration points is known for each.

| Stage | Floor shift error (median, p90) | Time on Graviton | Result |
|---|---|---|---|
| ECC screen, `findTransformECC` with a homography model on the still pixels | 4 mm, 11 mm | 228 ms | sends every move above half the tolerance to the second stage |
| ALIKED + LightGlue, USAC MAGSAC | 2 mm, 5 mm | 3,046 ms | trust decision |
| Both, as the console runs them | | | **84 of 84** moves above 0.25 m flagged, **0** false alarms in 132 checks |

A rotation-plus-shift model measures rolls well but reads a 2% zoom or tilt as steady; the
homography model measures all three. On the console the
screen runs every 3 s and ALIKED + LightGlue run only when the screen sees motion, when trust is
being restored, and every 10 minutes. A screen that measures a shift beyond the tolerance
withdraws trust at once (on test E it does so for 84 of 84 moves above 0.25 m and for none of the
132 below); only ALIKED + LightGlue give trust back. On the live t4g.small a knock is flagged 2.7
to 4.9 s after it happens (three trials through the API).

### 6.8 Speed

Per-stage medians on the t4g.small, same frames, pip `opencv-contrib-python-headless` 5.0.0.93,
console stopped (`bench/results/graviton2_t4g_small.json`):

| Stage | Full size | Half size (the console) |
|---|---|---|
| reduce the frame (`resize`, area) | | 0.7 ms |
| lens undistortion (`remap`) | 23.1 ms | 6.3 ms |
| head detection (HSV, components) | 22.4 ms | 3.8 ms |
| DIS optical flow (medium, 960×720) | 71.0 ms | 71.0 ms |
| overlay `warpPerspective`, `imencode` JPEG | 1.7 ms, 3.7 ms | 1.7 ms, 3.7 ms |
| ECC camera screen, every 3 s | | 228 ms |
| **one camera-second** | **718 ms** | **528 ms (1.9 cameras in real time)** |

| DM-Count (SHA) on Graviton | `ENGINE_NEW` | `ENGINE_CLASSIC` |
|---|---|---|
| zone crop (576×720) | 9,482 ms | **1,728 ms** |
| whole frame at 960×720 | 16,057 ms | 2,979 ms |

On Graviton the classic DNN engine runs DM-Count 5.5 times faster than the new graph engine and
returns the same count; on the AMD Zen 5 desktop the new engine is faster (6.9 s against 9.5 s
for the whole frame at 1280 px on two threads). `density.DEFAULT_ENGINE` picks the engine by
architecture. With the classic engine and the zone crop, one camera with DM-Count every 5 s needs
873 ms per second on the t4g.small. In the live console an agent's `cross_check_count` takes
under 6 s and `check_camera` under 0.3 s. Measured with both models free to run at once on
2 vCPUs and 2 GB, DM-Count on the whole frame with the new engine, and the features check every
30 s, they took 75 s and 23 s.

### 6.9 The console

axe-core (WCAG 2.1 AA and best practice) reports no violations on the console at 1600 px and 390 px
wide, or on the evening report, including contrast of the tier badges, a level-one heading and
a scrolling record a keyboard can reach.

`tools/ui_smoke.py` drives the real console in headless Chromium: knock the camera, wait for the
untrusted tile and the recalibration request, restore, wait for a proposal on Zone C, approve,
and wait for the verification. Console errors fail the run. It checks what API tests cannot:
that the replay decodes without seeking, that a knocked camera is flagged while a proposal is
pending, that every forecast in an episode uses the same frozen history, and that the card a
steward approves is the one that verifies.

## 7. AWS deployment

- **Compute.** One t4g.small (AWS Graviton, Arm) on Amazon Linux 2023 in us-east-1, IMDSv2
  required, encrypted gp3 volume, 2 GB swap. Python 3.11 with the headless OpenCV 5.0.0 aarch64
  wheel; JuPedSim is compiled from its source distribution because PyPI has no aarch64 Linux
  wheel. ALIKED + LightGlue need about 1.6 GB while they run and DM-Count about 0.7 GB, so on
  this instance each of those calls runs in a fresh child process (`BR_ISOLATE_MODELS=1`) and
  the memory goes back to the system.
- **Model.** gpt-oss-120b through the Amazon Bedrock API endpoint (OpenAI-compatible), with a
  short-lived bearer token minted from the instance role.
- **One heavy model at a time.** ALIKED + LightGlue and DM-Count each run in a fresh child process
  (`jobs.py` imports only OpenCV and the two model wrappers) behind a semaphore, so the two never
  share the 2 GB of memory.
- **Interfaces.** The console, the evening report at `/report`, and a read-only Model Context
  Protocol endpoint at `/mcp` (Streamable HTTP, JSON responses, 120 calls a minute) whose tools
  are the agent's own measurement tools. Approvals exist only in the console.
- **Network and access.** Security group open on 80 and 443 only; no SSH key, administration
  through AWS Systems Manager. Caddy terminates TLS with a Let's Encrypt certificate for the
  sslip.io name of the Elastic IP and adds HSTS; the app sets `X-Content-Type-Options`,
  `X-Frame-Options` and `Referrer-Policy`.
- **Least privilege.** The instance role can call the Bedrock API endpoint, append to and read
  one log group, and read one bucket. It cannot delete or change log events.
- **Observability.** The hash-chained trace on disk and in CloudWatch Logs (30-day retention).
- **A record a rewrite cannot pass.** A hash chain alone stops edits to single lines, not someone
  who rewrites the whole file and recomputes every hash. Two further checks do. Every entry is
  signed with an Ed25519 key that is made once at setup, kept in `/etc/breathingroom` and is
  readable but not writable by the service; its public key opens every run of the trace and is
  served at `/api/trace/key`. And because the role can only append to the log group, the
  CloudWatch copy is a witness the instance cannot rewrite: the evening report compares the
  local entries with it and says how many match (`trace.compare_cloudwatch`). A unit test
  rewrites a trace without the key: the chain re-hashes intact, the signatures and the
  CloudWatch comparison both fail.
- **Who decides.** Outside the public demonstration (`BR_MODE=production`, the default) a
  decision needs a steward's PIN, and is recorded under that steward's name: accounts are named,
  PINs are stored as salted PBKDF2 hashes, and wrong PINs lock out a client after 5 tries in
  5 minutes (and everyone after 30). The demonstration controls (knock the camera, restart, speed)
  exist only in demo mode. The public demonstration runs with `BR_MODE=demo` and no accounts, so
  anyone can try the approval; the trace and the evening report mark those decisions "no PIN".
- **Repeatability.** `deploy/provision.sh` creates every resource idempotently,
  `deploy/redeploy.sh` builds a bundle, uploads it to S3 and runs `deploy/setup_instance.sh`
  through SSM.
- **Cost control.** The console pauses when no browser has polled it for 2 minutes, which stops
  both the CPU work and the model calls; a returning visitor gets a fresh evening.

## 8. Limitations

- **Pushing detection is shown on overhead video of one experiment series, not on arbitrary
  CCTV.** E1's head detector relies on the caps of the Jülich participants. On ordinary CCTV,
  density would come from DM-Count, measured at 10.9% error on oblique street scenes, but no
  rated pushing data from an oblique camera was available, so pushing on such views is untested.
  On the overhead view the pushing index from DM-Count density reached AUROC 0.767 (heads 0.798);
  at its Youden point it flags 72% of pushing and 25% of walking person-seconds, and 85% of
  pushing episodes; set to flag 10% of walking, it flags 12% of pushing (6.1b). The console runs
  the sensitive point and a person looks at every alert; a camera that misses pushing gives no
  warning of it, and no approval step can make up for that.
- The pushing threshold was set and tested on six runs of one experiment series, with one camera.
- The venue-wide rule weighs the zone and the waiting area before each restricted entrance, but
  those areas are declared by the venue, not measured: no camera watches the queue, and approach
  routes and alternative entrances are not modelled (6.4b). Its errors and queue factors were
  measured on one simulated venue and carried to the others.
- The minutes-to-danger forecast is tested on eight simulated street layouts. Its pedestrian model cannot pack
  people above about 4.5/m², so forecast peaks above that are not scored, and the twin's
  behaviour at exits shaped what the forecast gets wrong.
- The forecast assumes the inflow trend lasts two minutes. A sudden surge (a train arriving) is
  seen only once it reaches the zone: in simulation ACT came 24 to 37 s after such a surge began
  and 156 to 374 s before 4.5/m² (6.4c), about a minute less lead than for a ramping crowd.
- The live console replays one recording and simulates the other zone. It has not run on a live
  venue camera.

## 9. Responsible use

- **What is kept, and what is shown.** The measurements and the record hold people-mass and
  motion on a floor grid. No faces, identities, video or image are written to disk or to the
  record, and no person is tracked across cameras. Images are shown: the console serves the
  latest camera frame with the pushing field drawn on it, and each pushing proposal a crop of the
  spot. Both are images of people. They are held in memory only (the frame until the next one,
  the 24 newest crops for 15 minutes at most) and served with `no-store`. On the public
  demonstration anyone can view them; with `BR_MODE=production` the frame, the crops, the zone
  state, the record and the report need a steward signed in with their PIN (a signed, HttpOnly,
  SameSite=Strict cookie that lasts a shift), and `/mcp` a bearer token.
- **Operational resilience is prototype-level.** One t4g.small shows the pipeline fits a small
  Arm instance. A zone whose source stops producing data is marked lost on the console, is not
  investigated, and turns `/health` into a 503 for an outside monitor; there is no second
  instance, no failover, no sustained test with many cameras and concurrent episodes, and no
  retention policy beyond the log group's 30 days. A venue pilot would need all of them, and a
  data-protection assessment.
- **A person decides.** The tier is computed from measurements; the model proposes; a steward
  approves. A camera that cannot be trusted stops counting until a person recalibrates it.
- **Accountability.** Every proposal, decision and outcome is in a tamper-evident trace, and the
  evening report states whether the chain re-hashes intact, whether the signatures hold and how
  many entries match the CloudWatch copy before it lists anything. Each decision names the
  steward who made it.
- **Other agents read, people decide.** The MCP endpoint exposes measurements and forecasts only.
  It has no tool that proposes, approves or changes the venue.
- **Misuse.** The same numbers could be used to disperse a lawful gathering. The intended
  operator is venue safety staff working under a written purpose, and the catalogue of actions is
  set by the venue: gates, doors, stewards and announcements.

## References

- Helbing, Johansson, Al-Abideen (2007). Dynamics of crowd disasters: an empirical study. Phys.
  Rev. E 75, 046109.
- Alia, Maree, Chraibi (2023). A cloud-based deep learning framework for early detection of pushing
  at crowded event entrances. IEEE Access.
- Forschungszentrum Jülich pedestrian data archive: DOIs 10.34735/ped.2018.1, 10.34735/ped.2022.4,
  10.34735/ped.2021.9 (CC BY 4.0).
- Wang et al. (2020). Distribution Matching for Crowd Counting (DM-Count). NeurIPS.
- Sundararaman et al. (2021). Tracking pedestrian heads in dense crowds (CroHD). CVPR.
- Zhao et al. (2023). ALIKED. Lindenberger et al. (2023). LightGlue.
- Kroeger et al. (2016). Fast optical flow using dense inverse search. ECCV.
- Fruin (1971). Pedestrian planning and design. Still (2014). Introduction to crowd science.
