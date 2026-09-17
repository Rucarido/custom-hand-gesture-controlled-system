# Use cases: niche applications for the webcam pointer-plus-pinch-click prototype

Synthesis of ten domain-scout reports (2026-09-17). Each scout read the
prototype source, gathered web evidence, and proposed exactly one niche.
This file summarizes, ranks, and picks one winner. It is stand-alone:
for full evidence trails, see the originating scout reports.

## 1. Project snapshot: what the prototype does today

- **Pipeline:** commodity webcam (640x480) -> MediaPipe Hand Landmarker
  (VIDEO mode, 21 landmarks, `max_num_hands: 1`, confidence thresholds 0.7)
  -> index-fingertip (landmark 8) absolute cursor mapping with mirror mode
  and a central motion-area window (0.2-0.8 x/y) -> smoothing stack (EMA
  alpha 0.35, 5 px dead zone, jump-guard dropping teleports > 10% of screen)
  -> single **pinch-click** (thumb-tip/index-tip distance, hysteresis,
  4 stable frames, 400 ms cooldown) emitted as an OS left-click via
  pynput/pyautogui. Modes: `tracker` / `pointer` / `full`; `p` toggles mouse
  output, `q` quits. Key files: `src/main.py`, `tracker.py`, `pointer.py`,
  `gestures_click.py`, `click_lock.py`, `filters.py`, `overlay.py`,
  `config/settings.yaml`.
- **Bonus feature:** click-lock freezes the cursor after ~10 still frames
  (shake to unlock), so the pinch does not drag the pointer — valuable for
  shaky, guarded, or messy hands.
- **Hard limits:** point + single left-click ONLY — no drag, scroll,
  right-click, double-click, or typing; one hand, one user; needs a visible
  hand in decent light; CPU-heavy (MediaPipe + TF + OpenCV).
- **Decisive cross-cutting limit — gloves:** MediaPipe hand tracking is
  trained on bare hands and widely reported to degrade on gloved hands
  (upstream issues google/mediapipe#4870, #3091). Any niche that *requires*
  gloves must prove gloved tracking first; niches with bare (even dirty)
  hands are inherently safer bets. Several scouts independently converged
  on this constraint.

## 2. Domain proposals (one compact section each)

### 2.1 Health — scrubbed-surgeon PACS series-select + slice-page (task `gesture-health-h1`)

- **Niche / user / task:** Operating surgeon, scrubbed in, standing 1.5-2.5 m
  from a side-table webcam, self-navigates already-open PACS/radiology images
  on a non-sterile display: point at a series thumbnail, pinch-click to
  select, page slices via big Next/Previous buttons — replacing "ask the
  circulating nurse to click".
- **Why valid:** Cannot touch mouse/keyboard without re-scrubbing; verbal
  relay is slow and error-prone. A randomized crossover study (Wipfli et al.,
  PLoS ONE 2016) measured gestures beating verbal relay for OR image
  manipulation; a long lineage (Graetzel JAMIA 2004; Jacob PMC3715344; Leap
  Motion + Carestream; Kinect intraoperative comparisons) plus commercial
  precedent (GestSure sterile-mouse system) backs the pain.
- **Fit:** Coarse pointer + single clicks on large targets is exactly the
  prototype's vocabulary; short single-user sessions match its design.
- **Top risk:** Gloved (possibly bloody) hands under OR lighting may not
  track at usable rates — the whole niche hinges on one untested question;
  plus hospital IT lockdown of synthetic input and the need for a read-only,
  harmless-click safety case.
- **Key sources:** doi:10.1197/jamia.m2410; doi:10.1371/journal.pone.0153596;
  PMC3715344; PMC3672422; doi:10.1177/1553350615587992;
  doi:10.3390/app132111982.

### 2.2 Industry & manufacturing — CNC job-shop doc stepper (task `gesture-industry-i2`)

- **Niche / user / task:** Machinist in a 5-50-person job shop with
  oil/coolant-wet **bare** hands steps through setup sheets, drawings, and
  tool-offset pages on the shop PC beside the machine — point at large
  NEXT/PREV/tab buttons, pinch-click to advance.
- **Why valid:** Oil-mist ghost touches and unresponsive capacitive panels
  are documented field failures CNC builders redesign hardware over; voice
  fails at 80-90 dB shop noise; sealed hygienic HMIs cost 10-50x a webcam.
  Bare oily skin still tracks as skin — the one dirty-hands niche where
  hands stay bare, dodging the glove limit.
- **Fit:** Read-mostly, big-target, ~1 click per few seconds: matches the
  click rate, single-click vocabulary, and click-lock stabilization exactly.
- **Top risk:** Shop lighting/flicker and absolute-mapping dropouts teleport
  the cursor; must be fenced to the documentation PC (never machine-motion
  controls) with a sealed mouse as permanent fallback.
- **Key sources:** DMC HG touchless HMI; Alps Alpine AirInput panel; US
  patent US11774940; CNC oil-mist touchscreen case studies (touchscreenxyue,
  touchwoipc); IEEE VRW 2024 field-reliability study (doi:10.1109/vrw62533.2024.00045).

### 2.3 Accessibility — grip-free secondary pointer for RSI / grip fatigue (task `gesture-access-a3`)

- **Niche / user / task:** Office worker/student/retiree with mouse-arm RSI,
  De Quervain's, or wrist arthritis who can move a hand but cannot tolerate
  gripping or repeated button force; uses the webcam pointer for light
  click-light tasks (reading, browsing, media) during pain flares — a
  zero-cost *secondary* pointer, not a mouse replacement.
- **Why valid:** MSDs are the top US occupational illness ($45-54B/yr);
  pooled pain prevalence ~66% in one IT cohort; mouse-time duration tied to
  wrist/hand symptoms (NUDATA, n=6,943). Every ergonomic vendor sells on "no
  gripping / no click force". Dedicated assistive head mice cost $599-1,995;
  this is $0 on hardware the user owns. Adjacent evidence: touch-free
  gestures helped MS-with-dexterity-impairment users (Springer 2026);
  Camera Mouse evaluated for cerebral palsy AAC (PMC10927611).
- **Fit:** This group HAS hand mobility but LACKS grip tolerance — the mirror
  image of the prototype's shape; click-lock freeze compensates shaky,
  guarded movements.
- **Top risk:** The pinch itself can hurt thumb-base arthritis sufferers
  ("pinch pain paradox") — a dwell-click fallback is gating, not optional;
  plus mid-air shoulder fatigue ("gorilla arm") capping session length.
- **Key sources:** doi:10.1007/s10209-026-01355-2; PMC10927611;
  PMC10840111; Wiley 10.1002/ajim.20081; GlassOuse/head-mouse pricing.

### 2.4 Education — science-teacher bench-demo clicker (task `gesture-edu-e4`)

- **Niche / user / task:** Secondary-school science teacher running a live
  bench demo (titration, projected simulation) with gloved/wet/chemically
  messy hands starts, pauses, and advances the simulation or slide deck from
  the bench — no walk back to the laptop, no contaminated keyboard, no
  shouting at a student to click.
- **Why valid:** Teacher is anchored at the bench; the laptop sits meters
  away for cleanliness/safety. Gesture wins where hands are
  occupied-but-visible and untouchable-to-surfaces. Published lecture-gesture
  systems (IEEE CSCWD doi:10.1109/cscwd.2017.8066683) and Leap Motion
  classroom studies show real effects; post-pandemic hygienic-classroom
  demand is documented (BenQ, GoLive 2026).
- **Fit:** Large Run/Pause/Next buttons need only point + discrete click;
  fixed bench camera mitigates lighting variance; click-lock suits lecture
  pacing.
- **Top risk:** False clicks from expressive teaching gestures (talking with
  hands) mid-lesson; plus the glove caveat if hands are gloved rather than
  merely wet.
- **Key sources:** doi:10.1109/cscwd.2017.8066683;
  doi:10.1109/chiuxid54398.2021.9650610; Leap Motion classroom teaching
  study; Maekelae et al. CHI 2022 touchless interaction.

### 2.5 Automotive — bay-side service-manual kiosk (task `gesture-auto-a5`)

- **Niche / user / task:** Line technician mid-job, hands greasy but
  ungloved, pages a fixed bay-side monitor showing PDF service manuals,
  wiring diagrams, torque tables — point at large next/prev buttons and
  hyperlinked sections, pinch-click, without touching the screen or washing up.
- **Why valid:** "Hands covered in 10W-30, about to ruin a $500 tablet" is
  documented workshop pain; grease mistriggers capacitive screens; bays are
  too noisy for voice; lookup is visual-spatial ("that connector"), which
  voice navigates poorly. Stationary, parked, non-driving use sidesteps the
  Euro NCAP 2026 / NHTSA pushback that kills driver-while-driving gestures.
- **Fit:** Move + single click on large targets is the full vocabulary;
  click-lock helps tired hands; HUD overlay gives aiming feedback.
- **Top risk:** Grease changes skin appearance and bay fluorescents create
  specular highlights — detection with realistically dirty (not clean demo)
  hands is unvalidated; no scroll/zoom for long procedures or diagram detail.
- **Key sources:** BMW Gesture Control (7 Series 2015+) as complement-only
  precedent; Euro NCAP 2026 physical-controls scoring; NHTSA-2010-0053;
  workshop grease-tablet builds; US 9037354 / US 11216180 (rejected
  passenger-RSE alternative).

### 2.6 Smart home — messy-hands recipe stepper (task `gesture-home-h6`)

- **Niche / user / task:** Home cook mid-recipe with flour-, dough-, oil-,
  or raw-meat-covered hands advances/goes back one recipe step and taps the
  single large on-screen control (next step, timer start/stop) on a
  counter-mounted tablet — from 0.5-1 m, without touching the screen.
- **Why valid:** ~one-third of home cooks touch smart devices after handling
  raw meat without washing (safefood 2024); FDA warns against phones/tablets
  as kitchen contamination vectors. Voice, the incumbent hands-free channel,
  drowns under extractor/sizzle noise (ACM CHI 2024 breakdowns); knobs have
  the same contamination problem as touch. Purpose-built precedent exists
  (GestureChef); Samsung ships camera gestures on Smart TVs, validating the
  interaction model.
- **Fit:** Recipe apps already present large Next/Back/Timer buttons; one
  deliberate click per 10-60 s matches the hysteresis + 4-frame + 400 ms
  cadence; click-lock holds the cursor for stiff messy hands; zero app
  integration (OS cursor).
- **Top risk:** Landmark accuracy with flour/dough/oil-covered fingers is
  untested; dim warm downlights/backlit windows can drop tracking — both
  testable in one kitchen session.
- **Key sources:** safefood 2024 "Smart Devices in the Kitchen"; FDA
  food-safety guidance; GestureChef (Hackster/Electromaker); ACM CHI 2024
  voice-breakdown study (DOI 10.1145/3613904.3642183); Samsung Gesture
  Interaction support pages.

### 2.7 Retail kiosk — hygienic QSR self-order point-and-confirm (task `gesture-retail-r7`)

- **Niche / user / task:** Walk-in quick-service-restaurant customer, one at
  a time, ~50-70 cm from a fixed camera beside existing kiosks, builds a
  meal order by pointing at big menu tiles and pinch-confirming each choice
  through a single confirm screen; payment stays at counter/card-tap.
- **Why valid:** Shared QSR touchscreens are a known contagion concern with
  stated customer preference for touchless where offered; chains already
  spend here — PepsiCo/KFC Poland gesture-ordering pilot (2021), Samsung
  Canada + St-Hubert AIRxTOUCH rollout, Ultraleap touchless self-order demo
  and UST Vision Checkout. Academic work backs the hygiene premise
  (arXiv:2107.05408; PalmSpace IJHCS 2024).
- **Fit:** Category -> item -> option -> confirm on oversized tiles is pure
  point-and-confirm; OS-cursor emulation retrofits onto the existing menu
  web app with zero integration.
- **Top risk:** Mid-air pointing historically underperforms touch on
  precision/usability — survives only with oversized targets and few steps;
  plus multi-person background confusion (single-hand model grabs the wrong
  hand), lighting variance, and pinch fatigue ("gorilla arm") at kiosk
  throughput.
- **Key sources:** PepsiCo/KFC Poland pilot (FoodIngredientsFirst);
  Ultraleap self-order gallery; Samsung Newsroom Canada (St-Hubert);
  Kiosk Marketplace (UST Vision Checkout); imageHOLDERS touchless totems.

### 2.8 Security & lab — gloved biosafety-cabinet protocol stepper (task `gesture-secure-s8`)

- **Niche / user / task:** Bench researcher/technician in nitrile gloves at
  a Class II biosafety cabinet steps a wall-mounted monitor through the
  day's paginated SOP checklist (reagent volumes, timers, decon steps) —
  NEXT/BACK/CHECK on 2-3 huge buttons, never touching shared surfaces.
- **Why valid:** Touching shared keyboards with contaminated gloves breaks
  containment; doff-use-PC-reglove wastes time and breaks aseptic flow —
  the GestSure OR economics at lab scale. Touch is the fomite; cabinet
  blower hum defeats voice; the mouse is exactly what must not be touched.
  Surveillance-video-wall operation was explicitly rejected (needs
  drag/zoom/multi-select, zero touchless-deployment evidence).
- **Fit:** 2-3 huge buttons + single click is the prototype's full envelope;
  dwell-lock guards accidental page turns.
- **Top risk:** The core interaction requires gloved hands — the single
  worst-known failure mode of the tracker (nitrile glare, untested); needs
  biosafety-officer sign-off on camera placement/airflow.
- **Key sources:** GestSure commercial system; doi:10.1016/j.csbj.2024.05.006
  (OR touchless review); doi:10.1177/1553350615587992; RGCB BSL-3 SOP / ABSL-4
  procedures; PMC5092084.

### 2.9 Entertainment — solo streamer stay-in-frame scene switcher (task `gesture-fun-f9`)

- **Niche / user / task:** Solo desk livestreamer (art, music practice,
  just-chatting) who is talent + camera operator + producer at once fires
  2-4 large, infrequent, non-time-critical scene/overlay switches
  (Main <-> Close-up <-> BRB, lower-third) with a raised-hand pinch while
  staying framed and talking — no reaching off-camera, no $100-200+ Stream
  Deck.
- **Why valid:** Reaching for hotkeys visibly breaks presence; foot pedals
  are awkward seated. Performers already do this with costlier hardware
  (Maskeliade's LEAP Motion + Ableton + Resolume show; djay camera gesture
  control; Roboflow's OBS gesture controller "act as your AI producer").
- **Fit:** Webcam already faces the user; 2-4 giant buttons need only
  point + deliberate click; dwell-lock parks the cursor safely while
  performing; HUD gives operator state.
- **Top risk:** False on-air scene switches from talking with hands near the
  face — the highest-impact failure mode; mitigated by
  require-index-extended + lock-by-default + cooldown, then measured.
- **Key sources:** Roboflow OBS gesture controller + repo; StreamGeeks
  gesture-OBS playground; Algoriddim djay Gesture Control; Ableton/Maskeliade
  interview; DJESTHESIA mouse-mapping precedent.

### 2.10 Field work — crop-scout dirty-hands paging clicker (task `gesture-field-f10`)

- **Niche / user / task:** Greenhouse/row-crop scout with soil-/sap-covered
  hands advances a tailgate-laptop scouting app (FarmQA / Crop-Scanner style:
  one record per plant; Next / Flag / Confirm buttons) with a quick
  bare-hand pinch-click from ~50 cm instead of touching the screen —
  ~one click per plant, hundreds per round.
- **Why valid:** Glove-off + wipe + smudge-screen every plant is the
  per-plant tax; voice fails in glasshouse fan noise; rugged-tablet vendors
  sell "glove/wet touch" as a premium feature, confirming the baseline pain.
  Chosen over construction plan-review (which needs absent pinch-zoom/drag/
  annotate).
- **Fit:** Coarse pointer + single click on big buttons; 400 ms cooldown
  rate-limits double-fire; `p` toggle is a credible muddy-hands safety.
- **Top risk:** Glasshouse harsh backlight + shade rows will drop the
  0.7-confidence RGB tracker; gusty outdoor reacquire causes cursor jumps;
  glove-on tracking untested (pick assumes a bare-hand flash); zero
  pinch-hysteresis band in current config (on == off == 0.06) risks flutter.
- **Key sources:** RockTECH/Winmate/CDTech rugged touch analyses; 3Rtablet
  agri tablet; ASCE 2025 construction-HRC gesture paper; BioBest
  Crop-Scanner, FarmQA, Mitti, SOFT.FARM workflow evidence.

## 3. Comparison ranking

Criteria (each scored 1-3, higher is better for the project):

- **Evidence strength:** peer-reviewed / deployed-commercial / quantified pain = 3;
  vendor docs + field reports = 2; adjacent-study extrapolation = 1-2.
- **Prototype fit:** today's point-plus-pinch on big targets with bare hands = 3;
  fits with one caveat (mess, noise, cadence) = 2; needs unproven tracking
  (gloves) or missing verbs = 1.
- **Validation cost (inverse):** half-day, no gatekeepers, household hardware = 3;
  needs site access or recruits = 2; needs regulated access (OR, BSL) = 1.
- **Risk (inverse):** false click harmless, fallback trivial = 3; embarrassment
  or distrust = 2; safety/regulatory/contamination consequences = 1.

| Rank | Domain | Niche | Evid. | Fit | Cost | Risk | Total |
| ------ | -------- | ------- | ------- | ----- | ------ | ------ | ------- |
| 1 | Smart home | Messy-hands recipe stepper | 3 | 3 | 3 | 3 | **12** |
| 2 | Industry | CNC job-shop doc stepper | 2 | 3 | 3 | 2 | **10** |
| 2 | Entertainment | Solo-streamer scene switcher | 2 | 3 | 3 | 2 | **10** |
| 4 | Accessibility | RSI grip-free secondary pointer | 3 | 2 | 3 | 2 | **10** |
| 5 | Retail | QSR self-order kiosk | 3 | 2 | 2 | 2 | **9** |
| 5 | Education | Bench-demo clicker | 2 | 2 | 3 | 2 | **9** |
| 7 | Automotive | Bay service-manual kiosk | 2 | 2 | 3 | 2 | **9** |
| 8 | Field | Crop-scout paging clicker | 2 | 2 | 2 | 1 | **7** |
| 9 | Security/lab | Biosafety protocol stepper | 2 | 1 | 1 | 2 | **6** |
| 9 | Health | OR PACS navigator | 3 | 1 | 1 | 1 | **6** |

Notes on the ordering:

- **Kitchen first:** the only niche scoring top marks on all four axes —
  quantified public-health pain + a shipped purpose-built precedent, a task
  shaped exactly like the prototype's two verbs at its natural cadence (one
  deliberate click per 10-60 s, false-click tolerant), a half-day validation
  in any home kitchen, and zero safety/regulatory blast radius.
- **CNC / streamer / RSI tied second:** CNC is the strongest *industrial*
  story (bare oily hands dodge the glove trap); the streamer is the cheapest
  validation of all (one evening, desk, no recruits); RSI has the largest
  pain pool but the pinch-pain paradox caps its fit score.
- **Retail/education/automotive next:** all valid, all slightly discounted —
  retail on precision/throughput risk and vendor competition; education on
  mid-lesson misfire risk; automotive as near-duplicate of the CNC niche
  (if CNC validates, auto follows nearly free).
- **Lab/OR last despite strong evidence:** the OR has the *best academic
  evidence of any domain*, but every point of it is offset by gloved-hand
  tracking risk, OR access cost, IT lockdown, and safety-critical
  consequences — exactly the wrong first bet for a prototype whose decisive
  unknown is "does it track real hands in real conditions". Validate bare
  hands first (kitchen/CNC), then spend the glove test where the evidence
  justifies it (OR second, lab third).
- **Field second-to-last:** harshest environment (backlight, gusts, dust,
  vibration) against an RGB-only tracker with a zero-width
  pinch-hysteresis band — outscores lab/OR on validation cost only because
  a tailgate test is easier to stage than a regulated bench or OR.

## 4. Overall recommended pick

**Smart-home messy-hands recipe step control ("flour-hands next-stepper",
task `gesture-home-h6`).**

A home cook with contaminated hands advances recipe steps and starts/stops
one timer on a counter-mounted tablet using only point-at-big-button +
pinch-click. It is real (food-safety-quantified, built by others, voice
proven to fail in the same room), it fits today's vocabulary with no code
changes, it fails fast on the honest unknowns (messy-hand tracking, kitchen
light), and a failure costs nothing while a success demos in one sentence:
*"advance a recipe with flour-covered hands."*

**First validation step (half day, no prototype changes):** mount a laptop at
a kitchen counter, open any recipe page at ~150% zoom (targets >= ~120 px),
have 1-2 cooks prepare something floury using only point + pinch-click for
3+ step-advances plus one timer start each; log successful advances, false
clicks, tracking losses, and time vs. the wash-hands baseline. **Go bar:**
>= 90% intended advances succeed first-try, zero screen touches, cooks prefer
it to washing-per-step. Run the messy-hand matrix inside the same session
(dry flour / wet dough / oil sheen / disposable glove) to rank which
conditions hold tracking and which need a detection profile.
