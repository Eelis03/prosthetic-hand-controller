# Prosthetic Hand Controller

Grasp taxonomy, proportional myoelectric control, and a simulated force feedback loop.

[![CI](https://github.com/Eelis03/prosthetic-hand-controller/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/prosthetic-hand-controller/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Grip force and slide against time for a plastic bottle. The detector fires 7 ms after the slide starts, the demand doubles from 1.20 N to 2.60 N, and the slide is arrested after 73 ms and 3.67 mm. The same object with the response switched off is out of the hand 96 ms after the lift.](docs/figures/slip_recovery.png)

## What it holds

A hand prosthesis is judged by whether it holds the cup. Ten objects, each in the
grasp of the taxonomy that suits it, are closed on, lifted, and made to take up
their own weight against a fingertip friction coefficient the controller is never
told. Nine are held.

`uv run python examples/grasp_evaluation.py`:

```
object          grasp            t_contact  t_grip   req_N   cmd_N  peak_N  over_%  slip_mm recov_ms  drop_ms         outcome
-----------------------------------------------------------------------------------------------------------------------------
drinking_glass  medium_wrap          0.161   0.210   1.525   2.600   2.600    0.00     0.46       28      n/a         success
plastic_bottle  medium_wrap          0.212   0.294   2.125   2.600   2.600    0.00     3.67       73      n/a         success
foam_cup        medium_wrap          0.212   0.442   0.028   1.200   1.200    0.00     0.00      n/a      n/a         success
paper_cup_full  medium_wrap          0.174   0.324   1.401   2.600   2.600    0.00     0.69       40      n/a         success
apple           power_sphere         0.147   0.208   1.471   2.600   2.600    0.00     0.64       34      n/a         success
hardback_book   prismatic_four       0.540   0.595   2.206   2.600   2.600    0.00     3.96       81      n/a         success
battery_cell    palmar_pinch         0.353   0.405   2.288   2.600   2.600    0.00     4.58       96      n/a         success
pen             tripod               0.810   0.879   0.112   1.200   1.200    0.00     0.00      n/a      n/a         success
door_key        lateral              0.646   0.716   0.163   1.200   1.200    0.00     0.00      n/a      n/a         success
steel_ball      power_sphere         0.179   0.225  19.221   2.600   2.592    0.00    20.35      n/a       66         dropped

trials              10
successes           9
success rate        90.0 percent
mean time to contact 0.343 s
mean force overshoot 0.00 percent
trials that slipped 7
trials dropped      1
mean slip recovery  58.7 ms
largest slip, held  4.58 mm
```

`req_N` is the grip force per contact needed to hold the object against gravity,
which the controller cannot compute because it knows neither the mass nor the
friction coefficient. `cmd_N` is what it settled on from the tactile signal
alone. Seven objects start to slide the moment they take up their weight,
because the nominal 1.20 N is below what they need, and six of the seven are
recovered: the object moves 0.46 mm to 4.58 mm and stops, in 28 ms to 96 ms,
58.7 ms on average. No held object comes near its crush limit. The water filled
paper cup peaks at 2.600 N against a limit of 4.000 N, and the foam cup at
1.200 N against 2.000 N.

## The one it drops

The steel ball is a 62 mm solid polished sphere weighing 0.980 kg with a friction
coefficient of 0.10 against the fingertip, held in a power sphere on five
contacts. From the same script:

```
Failure case: steel_ball
object              steel_ball
grasp               power_sphere (Power Sphere)
required force      19.221 N
commanded force     2.600 N
peak force          2.592 N
slip detections     1
peak slip speed     588.23 mm/s
total slip          20.35 mm
drop distance passed 0.066 s after the lift
force saturated     False
outcome             dropped
explanation         holding this object needs 19.22 N per contact, which is 1.28 times the 15.0 N safety limit
```

This object was never holdable. Five contacts at a friction coefficient of 0.10
would need 19.221 N each to carry a weight of 9.611 N, which is 1.28 times the
15.0 N per contact safety limit, and the limit exists because a hand that
squeezes harder than that damages what it is holding and overloads its own
actuator. The correct answer for this object is to refuse it.

What the trace adds is that the loop never even gets to argue. Slip is detected
6 ms after the ball starts to move, one response doubles the demand from 1.20 N
to 2.600 N, and the next response is barred for 150 ms by the refractory
interval that stops one slip episode from being counted many times. The ball has
slid 20.35 mm through the fingers and left the hand 0.066 s after the lift, less
than half way through that interval. A ladder that climbs a factor of two every
150 ms cannot reach 19.221 N in 66 ms from anywhere.

## Installation

Requires Python 3.12 or later. CI runs the whole suite on 3.12 and 3.13, on
Linux and on Windows, so the version floor in `pyproject.toml` is a tested claim
rather than a declared one.

```bash
git clone https://github.com/Eelis03/prosthetic-hand-controller.git
cd prosthetic-hand-controller
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running it

One grasp trial, and what happened in it:

```python
from hand_controller import reference_trial, simulate, summarise

trace = simulate(reference_trial("plastic_bottle"))
report = summarise(trace)

print(report.required_force)  # 2.1247741666666666 N
print(report.final_command)  # 2.6 N, after one slip response
print(report.total_slip)  # 0.0036666003345815702 m
print(report.slip_recovery_time)  # 0.073 s
print(report.success)  # True
```

The proportional control law and its latency:

```python
from hand_controller import ProportionalConfig, command_latency, control_law

config = ProportionalConfig()
print(control_law(config, 0.05))  # 0.0, inside the dead zone
print(control_law(config, 0.40))  # 0.7333333333333335 closure per second
print(control_law(config, 0.90))  # 1.25, saturated
print(command_latency(config, 0.001))  # 0.047 s against a 0.100 s budget
```

The static contact equilibrium of a rigid object against a deformable one:

```python
from hand_controller import default_hand, equilibrium_force, grasp, opposition_span
from hand_controller.model import default_pad

hand, pad = default_hand(), default_pad()
wrap = grasp("medium_wrap")
print(opposition_span(hand, wrap, 0.0))  # 0.0722197196034674 m
print(opposition_span(hand, wrap, 1.0))  # 0.02772522459903626 m
print(equilibrium_force(0.061, 0.065, pad, 5.0e7))  # 17.233087623506012 N
print(equilibrium_force(0.061, 0.065, pad, 8.0e3))  # 0.6061472451217309 N
```

Every table in this README is the output of one of these scripts:

```bash
uv run python examples/hand_kinematics.py
uv run python examples/grasp_taxonomy.py
uv run python examples/proportional_control.py
uv run python examples/mode_switching.py
uv run python examples/force_control.py
uv run python examples/slip_recovery.py
uv run python examples/grasp_evaluation.py
```

Each script that draws accepts `--no-figures`, and each script that simulates
accepts a reduced `--duration` for quick runs. Working figures go to `figures/`,
which is not tracked.

The two figures embedded above and below are different: they are tracked in
`docs/figures`, and one command rewrites both.

```bash
uv run python examples/readme_figures.py
```

They are snapshots of a run rather than fixtures, and nothing in CI compares
them, because Matplotlib output is not byte reproducible across platforms or
versions and a byte comparison would fail on a font metric rather than on
anything about the controller. The script reports the total size and fails if
the pair leaves the 250 kB budget, which is the property that actually matters
for a tracked binary.

## The control stack

The hand is five serial chains of three segments each, sized from the
anthropometric measurements of Buchholz, Armstrong and Goldstein (1992). Each
finger carries one actuator, and its two distal joints follow the base joint
through fixed ratios, which is how an underactuated prosthetic finger transmits
one motor through a linkage (Birglen et al., 2008; Belter et al., 2013). Ten
commanded numbers expand to fifteen joint angles. The thumb has two degrees of
freedom, flexion and opposition, where opposition rotates the thumb flexion
plane from across the palm to palmar so that the thumb pad faces the finger pads.

Six grasps are implemented as data, taken from the GRASP taxonomy of Feix,
Romero, Schmiedmayer, Dollar and Kragic (2016) and classified on the power to
precision axis of Cutkosky (1989). Each records its taxonomy number, its
opposition type, its thumb position, its open and closed joint postures, and the
digits expected to contact the object. The contact expectation is what turns a
posture into a mechanical prediction, because it fixes how many surfaces share
the tangential load.

The user interface is the conventional two site arrangement reviewed by Fougner
et al. (2012): the difference of two activation envelopes passes through a dead
zone, a gain and a saturation to produce a closing velocity. That is one degree
of freedom for a hand that has ten, so everything else has to come from the
controller. Grasp selection uses co-contraction, and the recogniser requires the
two envelopes to be both active and balanced, which is what separates a genuine
co-contraction from a strong single site effort with electrode crosstalk.

Contact is Hertzian, with the fingertip pad and the object treated as two springs
in series and the energy loss added as the velocity proportional term of Hunt and
Crossley (1975). The series stiffness is dominated by the softer body, which is
the fact the force loop has to survive: the same commanded opening produces
17.233 N on a glass and 0.246 N on foam.

The grip force loop follows the phase structure Romano et al. (2011) use for a
tactile robotic grasp controller. Close under the user's command, detect contact,
ramp to a light nominal force, then hold and raise the force only when slip is
reported. Slip is detected from the band limited energy of the tactile signal,
following the stress rate sensing of Howe and Cutkosky (1993), because a sensor
that responds to the rate of change of contact stress sees the vibration of a
sliding contact long before the static force reveals anything.

| Module | Responsibility |
| --- | --- |
| `model/anatomy.py` | Link lengths, joint ranges, pad radii, and the underactuated joint coupling |
| `model/kinematics.py` | Forward kinematics of every digit, thumb opposition, and joint limit checking |
| `model/grasps.py` | The grasp taxonomy as data, the opposition span, and the closure solver |
| `model/objects.py` | Object properties and whether a grasp can enclose one |
| `model/contact.py` | Hertzian series stiffness, Hunt-Crossley force, contact detection, friction capacity |
| `algorithm/protocols.py` | The control law, force regulator and slip detector interfaces the loop depends on |
| `algorithm/proportional.py` | Dead zone, gain and saturation, envelope smoothing, and the latency measurement |
| `algorithm/modeswitch.py` | Co-contraction recognition, grasp cycling, and the complete two site front end |
| `algorithm/force.py` | Grip force regulation with an online plant gain estimate, and the safety clamp |
| `algorithm/slip.py` | Band pass, rectify, smooth and threshold slip detection |
| `pipeline/emg.py` | Simulated activation envelopes described as trapezoidal bursts |
| `pipeline/simulation.py` | The closed loop, the tactile sensor model, and the recorded trace |
| `pipeline/scenarios.py` | Named configurations shared by the examples, the tests and the regression file |
| `analysis/metrics.py` | Success verdicts, timings, force statistics, and slip episodes |
| `analysis/report.py` | Fixed width text rendering of every result table |
| `analysis/figures.py` | Every figure, and the only module that imports Matplotlib |

The layers depend downward only. `model` imports nothing from the package,
`algorithm` imports only its own protocols, `pipeline` imports both, and
`analysis` imports all three. Nothing in `model` or `algorithm` performs input or
output. The closed loop reaches the force regulator and the slip detector only
through the protocols in `algorithm/protocols.py`, so either can be replaced
without touching the loop. The alternatives that were considered and rejected,
including pattern recognition control, an integral term in the force loop, and a
full rigid body grasp simulation, are recorded in
[docs/design-notes.md](docs/design-notes.md).

## Results

Everything below is the output of the command shown above it. The configuration
is the reference hand, a control period of 1 ms, a trial of 3.0 s with the object
taking up its own weight at 1.0 s, a nominal grip force of 1.20 N, a safety limit
of 15.0 N per contact, and a slip response that doubles the demand and adds
0.20 N. A trial ends early if the object slides 20 mm, which is the distance at
which it has left the hand and there is nothing further to simulate.

### The hand and its grasps

`uv run python examples/hand_kinematics.py`:

```
Segment lengths, millimetres
digit       proximal    middle    distal     total
--------------------------------------------------
index           39.8      22.4      15.8      78.0
middle          44.6      26.3      17.4      88.3
ring            41.4      25.7      17.3      84.4
little          32.7      18.7      15.8      67.2
thumb           46.2      31.6      21.1      98.9

Finger coupling: pip = 1.0000 x mcp, dip = 0.6667 x pip
Thumb coupling:  mcp = 0.8000 x cmc, ip = 0.6667 x mcp

Span of each grasp and joint limit check over the whole closure sweep
grasp             closed_mm   open_mm  violations  monotone
-----------------------------------------------------------
medium_wrap           27.73     72.22           0      True
power_sphere         -14.28     71.22           0      True
prismatic_four         6.50     72.43           0      True
palmar_pinch         -18.05     49.73           0      True
tripod                 6.75     60.20           0      True
lateral               -4.15     27.11           0      True
```

Every grasp is inside its joint limits at all 21 sampled points of its closing
trajectory, and every span falls strictly, so an object is met once rather than
twice. A negative closed span means the two opposing pads would overlap, that is,
the grasp can close on nothing. The index chain reproduces its own segment
lengths to 3.469e-18 m, which is the rounding of a sum of three numbers of order
0.04 m.

![The Medium Wrap drawn from the side at three object widths. The same coupled finger trajectory is stopped by the object at closure 0.03 on a 70 mm width, 0.32 on 50 mm and 0.76 on 30 mm, so one commanded number and a fixed coupling produce the whole posture.](docs/figures/grasp_postures.png)

The figure is what the span table cannot show: the shape the fifteen joint angles
take, and how far along one trajectory each object stops the hand. The thumb, in
red, is foreshortened because its opposition carries it out of the plane the
fingers flex in, which is a property of this thumb model and is recorded under
its limitations.

`uv run python examples/grasp_taxonomy.py`:

```
grasp           no taxonomy name         opposition  thumb      class         contacts  closed_mm  open_mm
----------------------------------------------------------------------------------------------------------
medium_wrap      3 Medium Wrap           palm        abducted   power                6       27.7     72.2
power_sphere    11 Power Sphere          palm        abducted   power                5      -14.3     71.2
prismatic_four   6 Prismatic Four Finger pad         abducted   precision            5        6.5     72.4
palmar_pinch     9 Palmar Pinch          pad         abducted   precision            2      -18.0     49.7
tripod          14 Tripod                pad         abducted   precision            3        6.8     60.2
lateral         16 Lateral               side        adducted   intermediate         2       -4.2     27.1
```

The six cover palm, pad and side opposition, both thumb positions, and all three
power classes. The `contacts` column is the number of surfaces the taxonomy
expects to bear load, and it is what sets the friction capacity at a given grip
force: a medium wrap on six contacts carries three times the tangential load that
a two digit pinch carries at the same force. It is also why the battery cell,
which weighs 0.140 kg, needs more grip force per contact than the drinking glass,
which weighs 0.420 kg.

### Proportional control and its latency

`uv run python examples/proportional_control.py`:

```
Proportional control law
dead zone           0.100 of full activation
gain                2.200 closure per second per unit activation
saturation          1.250 closure per second
saturating at       0.6114 activation difference
points sampled      4001
maximum command inside the dead zone 0.000e+00
minimum command outside it           2.035e-16
monotone non decreasing              True
peak command                         1.2500

Latency
proportional path   47.0 ms of a 100 ms budget, met: True
mode switch         178.0 ms of a 250 ms budget, met: True
compute per period  4.9 us against a 1000 us period, real time factor 202
```

The characteristic is exactly zero at every sampled point inside the dead zone,
non zero at every point outside it, monotone across the whole range, and flat at
the configured saturation. The dead zone is applied with rescaling, so the map
has no step at its edge, which a user would feel as a lurch.

The 47.0 ms proportional latency is the 90 percent rise time of the commanded
velocity after a step in the closing envelope, measured end to end through the
50 ms smoothing window and the command slew limit rather than added up from
them. It sits inside the 100 ms to 125 ms optimum that Farrell and Weir (2007)
measured for controller delay. The 178.0 ms mode switch latency is the delay from
the onset of a co-contraction to the grasp changing, against a budget of 250 ms.
The throughput figure is machine dependent and is reported rather than asserted.

Grasp selection and its false trigger case, from the same script:

```
pattern                              open  close  recognised
------------------------------------------------------------
rest                                 0.02   0.02       False
strong close only                    0.05   0.90       False
strong close with crosstalk          0.45   0.90       False
strong open only                     0.90   0.05       False
balanced co-contraction              0.70   0.70        True
co-contraction, slight imbalance     0.60   0.70        True
weak balanced effort                 0.20   0.20       False

pattern                             switches  final velocity
------------------------------------------------------------
rest                                       0          0.0000
strong close only                          0          1.2500
strong close with crosstalk                0          0.8556
strong open only                           0         -1.2500
balanced co-contraction                    1          0.0000
co-contraction, slight imbalance           1          0.0000
weak balanced effort                       0          0.0000
```

The third row is the case that matters. A hard closing effort with 0.45 of
crosstalk on the opening channel has both electrodes above the activation
threshold, and a recogniser that tested only activation would change grasp in the
middle of a reach. Requiring the two envelopes to be balanced as well rejects it,
while still accepting the imbalance of 0.10 in the sixth row. The velocity column
shows the other half of the arrangement: a confirmed or refractory co-contraction
gates the proportional command to zero, so selecting a grasp never moves the hand.

`uv run python examples/mode_switching.py` drives the same front end from a
sequence of co-contraction bursts:

```
  burst_s  switch_s   lag_ms             grasp
----------------------------------------------
     0.30     0.518    218.0      power_sphere
     0.90     1.118    218.0    prismatic_four
     1.50     1.718    218.0      palmar_pinch
switches                3
peak commanded velocity 0.0000 closure per second

The same front end driven by a strong single site contraction
peak closing envelope   1.000
peak opening envelope   0.052
switches                0
peak commanded velocity 1.2500 closure per second
```

The 218.0 ms lag is measured from the start of the burst rather than from a step,
so it includes the 80 ms rise of the contraction itself on top of the 178.0 ms
step response. A full second of maximum single site effort produces no switch at
all. Selecting among six grasps by cycling therefore costs up to five of these,
which is the price of a two site interface and is measured here rather than
hidden.

### Rigid against deformable contact

`uv run python examples/force_control.py`, with the span held 2.0 mm inside a
65 mm object and no control acting:

```
 stiffness_N/m^1.5       k_eff   force_N  travel_for_2N_mm
----------------------------------------------------------
             5e+07   1.927e+05    17.233             0.476   rigid glass
             8e+03        6777     0.606             4.433   wet paper
             3e+03        2746     0.246             8.096   foam
```

The same commanded span produces 17.233 N on the glass and 0.246 N on the foam, a
factor of seventy. This is the series stiffness at work: against a rigid object
the effective stiffness is 1.927e5, within 4 percent of the pad alone, while
against foam it is 2746, close to the object alone.

The same distinction inside the closed loop, from the same script:

```
object                 k_eff  indent_mm  closure  force_N
---------------------------------------------------------
drinking_glass     1.927e+05      0.567   0.0157    2.600
paper_cup_full          6777      5.280   0.1485    2.600
foam_cup                2746      5.759   0.1652    1.200
```

Reaching 2.600 N on the glass costs 0.567 mm of indentation and 1.57 percent of
the closing travel. Reaching the same force on the water filled paper cup costs
5.280 mm and 14.85 percent of the travel, nine times more. The loop reaches both
because its proportional gain is divided by an online estimate of how much force
one unit of closure buys, and that quantity differs by two orders of magnitude
between them. A fixed gain that settles the glass leaves the paper cup still
moving.

Force overshoot is 0.00 percent on every object, and the steady state error at
the end of these three trials lies between 7.550e-15 N and 9.104e-14 N. Neither
is a tuning achievement. The plant from closure rate to force already contains an
integrator, so a proportional law converges without offset, and the demand is
rate limited, so there is nothing for the loop to overshoot. An integral term
would add windup during the approach and buy nothing.

### What the slip response is worth

`uv run python examples/slip_recovery.py` runs the whole object set twice, with
the slip response enabled and disabled, against the same disturbance:

```
object            slip_on_mm  recov_on_ms  outcome_on  outcome_off  drop_off_ms
-------------------------------------------------------------------------------
drinking_glass          0.46           28     success      dropped          137
plastic_bottle          3.67           73     success      dropped           96
foam_cup                0.00          n/a     success      success          n/a
paper_cup_full          0.69           40     success      dropped          168
apple                   0.64           34     success      dropped          148
hardback_book           3.96           81     success      dropped           94
battery_cell            4.58           96     success      dropped           92
pen                     0.00          n/a     success      success          n/a
door_key                0.00          n/a     success      success          n/a
steel_ball             20.35          n/a     dropped      dropped           65
-------------------------------------------------------------------------------
objects held               9                                     3
```

Nine objects held with the slip response against three without it. The three that
survive without it are the three that never needed more than the nominal force.
Every other object, the steel ball included, is out of the hand within 65 ms to
168 ms of taking up its own weight. This is the measurement that gives the
response its value, and it is a
comparison rather than an argument: the same objects, the same disturbance, one
switch.

The timing of the response, from the same script:

```
object            onset_s  detect_s  detect_lag_ms  recovery_ms  slip_mm  peak_v_mm_s
-------------------------------------------------------------------------------------
drinking_glass      1.000     1.008            8.0           28     0.46        26.04
plastic_bottle      1.000     1.007            7.0           73     3.67        84.31
paper_cup_full      1.000     1.012           12.0           40     0.69        27.40
apple               1.000     1.009            9.0           34     0.64        30.10
hardback_book       1.000     1.006            6.0           81     3.96        83.72
battery_cell        1.000     1.006            6.0           96     4.58        84.13
steel_ball          1.000     1.006            6.0          n/a    20.35       588.23
```

Detection takes 6 ms to 12 ms from the first movement, on the scale of the 6 ms
envelope time constant plus the 4 ms confirmation. Recovery, measured from the
start of sliding to the end of it, takes 28 ms to 96 ms and costs 0.46 mm to
4.58 mm of sliding. The spread follows the size of the shortfall: the drinking
glass needed 1.525 N and had 1.200 N, so it barely accelerated, while the battery
cell needed 2.288 N on two contacts and slid 4.58 mm before the doubled demand
arrived. The steel ball has no recovery time because it never stopped.

## What the model does not do

This is a simulation. No electromyogram is recorded, no classifier is trained, no
hardware is driven, and no measurement of any kind comes from a physical hand.
Every number above is the output of the model, and the model is only as good as
its assumptions. The ones that matter most:

- The grasp is reduced to one span and one normal force. There is no grasp
  matrix, no friction cone per contact, and no torque about the object, so a
  grasp that would fail by rotating in the hand is reported as holding.
- Two site proportional control reaches few degrees of freedom, which is a real
  burden on the user that this project measures rather than hides.
- The tactile sensor is a model, not a device. The detector structure would
  survive contact with a real stress rate sensor; the thresholds would have to be
  recalibrated.
- Slip is detected after it starts, not before. Incipient slip is not modelled,
  so the object always moves some distance before the response arrives.
- The object set is ten objects chosen by hand. A success rate of 90.0 percent is
  a statement about these ten under this disturbance, not a benchmark.

The full list, with what each one costs and what removing it would take, is in
[docs/design-notes.md](docs/design-notes.md), which also records which limitation
was most recently closed and what closing it changed.

## Testing, coverage and reproducibility

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov=src/hand_controller --cov-report=term-missing
```

There are 2909 tests, the suite runs in about forty seconds, and statement
coverage of `src/hand_controller` is 98.56 percent. CI enforces
`--cov-fail-under=96`, which is the measured value rounded down less two, so that
a genuine regression fails the build and a one line refactor does not.

The suite has three tiers. The property and invariant tier covers the
mathematics: that forward kinematics reproduces the phalanx lengths, that the
joint coupling equals the declared ratios exactly rather than approximately, that
every grasp of the taxonomy is inside its joint limits and closes monotonically,
that the proportional law is zero at every one of 401 sampled points inside the
dead zone and non zero at every point outside it, that a strong single site
contraction never triggers a mode switch at any of 201 activation levels, that
contact detection fires when and only when the indentation is positive across
2001 points spanning the sign change, that injected slip is detected and the
resulting force increase stops it, and that the commanded force stays inside the
safety interval under a thousand adversarial inputs including infinities and
quiet not a number values.

The regression tier recomputes the whole evaluation set and compares it against
`tests/data/reference_run.json`. Which quantities are pinned was decided by
measurement, not by argument. Every candidate was recomputed with one object
property perturbed by a relative 1e-12 and by a relative 1e-9, across five
properties and every trial, and the movement that produced is that quantity's
reproducibility scale.

The result splits cleanly. Grasp geometry, every discrete verdict, the slip
detection count, `time_to_contact`, `time_to_grip`, `drop_time` and the settled
forces of a trial that kept its object do not move at all, or move only by
rounding at 7.0e-14 N, so they are pinned. `slip_recovery_time` moves by up to 19
control periods, `total_slip` by up to 2.08 mm, `peak_slip_speed` by up to 43.8
percent, and the forces of the trial that lost its object by up to 2.71e-2 N, so
none of those is pinned. All of them sit downstream of the finite difference the
force regulator uses to estimate its plant gain, which is a ratio of two small
differences and therefore ill conditioned by construction; in the first control
periods after a slip response it turns a 1e-12 change in the geometry into a
seven percent change in grip force. The force settles to the same value either
way, which is why the settled quantities are exact, but the deceleration differs
throughout the arrest and the instant the object stops moves by many samples.
Quantisation bounds a readout, not a crossing, and this crossing is reached
tangentially.

The unstable quantities are bounded instead, against configuration constants or
other pinned quantities rather than against recorded values, so that a bound
cannot drift towards whatever a run happens to produce. Recovery must complete
within the slip response refractory interval of 150 ms, against a largest
recorded value of 96 ms. The slide must stay under half the 20 mm drop distance,
against a largest recorded value of 4.58 mm. The forces of the trial that lost
its object must stay below the demand the loop was still chasing when it left,
which is pinned and exact. A test switches the slip response off to confirm that
the first two bounds are violated when the behaviour they protect is broken, and
two more
recompute the whole evaluation set under those same perturbations and assert that
every pinned field is unchanged and every bound still holds. Run
`uv run python tests/test_regression.py` to regenerate the reference file after a
reviewed change of behaviour.

The integration tier loads every script in `examples/`, runs it with a reduced
duration, and checks that it exits cleanly and writes what it says it wrote. A
separate test asserts that no example is missing from that table, and further
tests exercise every figure writing path, the presence of the `py.typed` marker
inside the package, and the size and alt text of the figures this README embeds.

## References

### Grasp taxonomy and hand anatomy

- Feix, T., Romero, J., Schmiedmayer, H.-B., Dollar, A. M. and Kragic, D. (2016).
  The GRASP taxonomy of human grasp types. *IEEE Transactions on Human-Machine
  Systems* 46(1), 66-77. DOI:
  [10.1109/THMS.2015.2470657](https://doi.org/10.1109/THMS.2015.2470657). Source
  of the six grasps implemented here, of their numbering, and of the opposition
  type and thumb position classification.
- Cutkosky, M. R. (1989). On grasp choice, grasp models, and the design of hands
  for manufacturing tasks. *IEEE Transactions on Robotics and Automation* 5(3),
  269-279. DOI: [10.1109/70.34763](https://doi.org/10.1109/70.34763). Source of
  the power to precision axis on which each grasp is classified.
- Buchholz, B., Armstrong, T. J. and Goldstein, S. A. (1992). Anthropometric data
  for describing the kinematics of the human hand. *Ergonomics* 35(3), 261-273.
  DOI:
  [10.1080/00140139208967812](https://doi.org/10.1080/00140139208967812). Source
  of the phalanx lengths and of the metacarpophalangeal joint positions.
- Rijpkema, H. and Girard, M. (1991). Computer animation of knowledge-based human
  grasping. *Proceedings of SIGGRAPH 91*, 339-348. DOI:
  [10.1145/122718.122754](https://doi.org/10.1145/122718.122754). Source of the
  two thirds relation between the distal and proximal interphalangeal joints.

### Hand mechanism design

- Birglen, L., Laliberte, T. and Gosselin, C. (2008). *Underactuated Robotic
  Hands*. Springer Tracts in Advanced Robotics 40. DOI:
  [10.1007/978-3-540-77459-4](https://doi.org/10.1007/978-3-540-77459-4). Source
  of the underactuated finger model in which one actuator drives three coupled
  joints.
- Belter, J. T., Segil, J. L., Dollar, A. M. and Weir, R. F. (2013). Mechanical
  design and performance specifications of anthropomorphic prosthetic hands: a
  review. *Journal of Rehabilitation Research and Development* 50(5), 599-618.
  DOI:
  [10.1682/JRRD.2011.10.0188](https://doi.org/10.1682/JRRD.2011.10.0188). Source
  of the closing time and pinch force ranges that set the saturation and the
  safety limit, and of the grasp set that commercial hands provide.
- Controzzi, M., Cipriani, C. and Carrozza, M. C. (2014). Design of artificial
  hands: a review. In *The Human Hand as an Inspiration for Robot Hand
  Development*, Springer Tracts in Advanced Robotics 95, 219-246. DOI:
  [10.1007/978-3-319-03017-3_11](https://doi.org/10.1007/978-3-319-03017-3_11).
  Background on transmission choices in prosthetic hands, discussed in the design
  notes.

### Myoelectric control

- Fougner, A., Stavdahl, O., Kyberd, P. J., Losier, Y. G. and Parker, P. A.
  (2012). Control of upper limb prostheses: terminology and proportional
  myoelectric control, a review. *IEEE Transactions on Neural Systems and
  Rehabilitation Engineering* 20(5), 663-677. DOI:
  [10.1109/TNSRE.2012.2196711](https://doi.org/10.1109/TNSRE.2012.2196711).
  Source of the two site proportional structure, of the dead zone, gain and
  saturation terminology, and of co-contraction as the mode switch signal.
- Parker, P., Englehart, K. and Hudgins, B. (2006). Myoelectric signal processing
  for control of powered limb prostheses. *Journal of Electromyography and
  Kinesiology* 16(6), 541-548. DOI:
  [10.1016/j.jelekin.2006.08.006](https://doi.org/10.1016/j.jelekin.2006.08.006).
  Source of the activation envelope as the signal a two site controller acts on.
- Farrell, T. R. and Weir, R. F. (2007). The optimal controller delay for
  myoelectric prostheses. *IEEE Transactions on Neural Systems and Rehabilitation
  Engineering* 15(1), 111-118. DOI:
  [10.1109/TNSRE.2007.891391](https://doi.org/10.1109/TNSRE.2007.891391). Source
  of the 100 ms latency budget for the proportional path.
- Scheme, E. and Englehart, K. (2011). Electromyogram pattern recognition for
  control of powered upper-limb prostheses: state of the art and challenges for
  clinical use. *Journal of Rehabilitation Research and Development* 48(6),
  643-659. DOI:
  [10.1682/JRRD.2010.09.0177](https://doi.org/10.1682/JRRD.2010.09.0177). Source
  of the comparison between two site and pattern recognition control that the
  design notes record.
- Englehart, K. and Hudgins, B. (2003). A robust, real-time control scheme for
  multifunction myoelectric control. *IEEE Transactions on Biomedical
  Engineering* 50(7), 848-854. DOI:
  [10.1109/TBME.2003.813539](https://doi.org/10.1109/TBME.2003.813539). The
  pattern recognition alternative that was considered and not implemented.
- Cipriani, C., Zaccone, F., Micera, S. and Carrozza, M. C. (2008). On the shared
  control of an EMG-controlled prosthetic hand: analysis of user-prosthesis
  interaction. *IEEE Transactions on Robotics* 24(1), 170-184. DOI:
  [10.1109/TRO.2007.910708](https://doi.org/10.1109/TRO.2007.910708). Source of
  the shared control arrangement in which the user commands closing and the
  controller regulates force.

### Contact, grip force and slip

- Hunt, K. H. and Crossley, F. R. E. (1975). Coefficient of restitution
  interpreted as damping in vibroimpact. *Journal of Applied Mechanics* 42(2),
  440-445. DOI: [10.1115/1.3423596](https://doi.org/10.1115/1.3423596). Source of
  the velocity proportional damping term that keeps the contact force continuous
  at touchdown.
- Johnson, K. L. (1985). *Contact Mechanics*. Cambridge University Press. DOI:
  [10.1017/CBO9781139171731](https://doi.org/10.1017/CBO9781139171731). Source of
  the Hertzian three halves power law and of the series compliance of two curved
  bodies in contact.
- Johansson, R. S. and Westling, G. (1984). Roles of glabrous skin receptors and
  sensorimotor memory in automatic control of precision grip when lifting rougher
  or more slippery objects. *Experimental Brain Research* 56(3), 550-564. DOI:
  [10.1007/BF00237997](https://doi.org/10.1007/BF00237997). Source of the finding
  that the human grip force is set just above the slip ratio, which is the
  strategy the nominal force implements.
- Johansson, R. S. and Westling, G. (1987). Signals in tactile afferents from the
  fingers eliciting adaptive motor responses during precision grip. *Experimental
  Brain Research* 66(1), 141-154. DOI:
  [10.1007/BF00236210](https://doi.org/10.1007/BF00236210). Source of the grip
  force upgrade that follows a slip signal, and of its latency.
- Howe, R. D. and Cutkosky, M. R. (1993). Dynamic tactile sensing: perception of
  fine surface features with stress rate sensing. *IEEE Transactions on Robotics
  and Automation* 9(2), 140-151. DOI:
  [10.1109/70.238278](https://doi.org/10.1109/70.238278). Source of the band
  limited vibration signature of slip and of the sensing band the detector uses.
- Romano, J. M., Hsiao, K., Niemeyer, G., Chitta, S. and Kuchenbecker, K. J.
  (2011). Human-inspired robotic grasp control with tactile sensing. *IEEE
  Transactions on Robotics* 27(6), 1067-1079. DOI:
  [10.1109/TRO.2011.2162271](https://doi.org/10.1109/TRO.2011.2162271). Source of
  the phase structure of the grasp, of the light nominal force, and of the
  practice of raising the force only in response to tactile evidence.
- Engeberg, E. D. and Meek, S. G. (2013). Adaptive sliding mode control for
  prosthetic hands to simultaneously prevent slip and minimize deformation of
  grasped objects. *IEEE/ASME Transactions on Mechatronics* 18(1), 376-385. DOI:
  [10.1109/TMECH.2011.2179061](https://doi.org/10.1109/TMECH.2011.2179061). The
  adaptive slip prevention alternative that the design notes record as rejected.
- Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*.
  Prentice-Hall. ISBN: 978-0-13-022335-7. The root finding method behind
  `scipy.optimize.brentq`, which inverts the grasp span.

### Dependencies

| Package | Purpose | Licence |
| --- | --- | --- |
| [NumPy](https://numpy.org/) >= 2.0 | Trace storage, the vector algebra in the kinematics, and the seeded generators behind the simulated envelopes and the sensor noise | BSD-3-Clause |
| [SciPy](https://scipy.org/) >= 1.14 | `scipy.signal.butter` and `scipy.signal.sosfilt` for the slip detection band pass, and `scipy.optimize.brentq` for the grasp span inverse | BSD-3-Clause |
| [Matplotlib](https://matplotlib.org/) >= 3.9 | Every figure, including the two tracked in `docs/figures` | Matplotlib licence, a BSD compatible PSF style licence |
| [pytest](https://pytest.org/) >= 8.3 | Test runner for all three test tiers | MIT |
| [pytest-cov](https://pytest-cov.readthedocs.io/) >= 6.0 | Statement coverage, enforced in CI | MIT |
| [Ruff](https://docs.astral.sh/ruff/) >= 0.8 | Linting and import ordering | MIT |
| [mypy](https://mypy-lang.org/) >= 1.13 | Static type checking of `src/hand_controller` under strict settings | MIT |

Dependency citations:

- Harris, C. R. et al. (2020). Array programming with NumPy. *Nature* 585,
  357-362. DOI:
  [10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2).
- Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific
  computing in Python. *Nature Methods* 17, 261-272. DOI:
  [10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2).
- Hunter, J. D. (2007). Matplotlib: a 2D graphics environment. *Computing in
  Science and Engineering* 9(3), 90-95. DOI:
  [10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55).

## License

Released under the MIT license. See [LICENSE](LICENSE).
