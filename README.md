# Prosthetic Hand Controller

Grasp taxonomy, proportional myoelectric control, and a simulated force feedback loop.

[![CI](https://github.com/Eelis03/prosthetic-hand-controller/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/prosthetic-hand-controller/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This library simulates the control loop of a multi-articulating myoelectric hand
prosthesis from the moment the user starts to close it to the moment the object
is secure. It contains a fifteen joint kinematic model of an underactuated hand,
six grasps taken from the GRASP taxonomy of Feix et al. (2016), a two site
proportional myoelectric front end with co-contraction grasp selection, a
compliant contact model, and a grip force loop that detects slip from a tactile
signal and responds to it. It is intended for engineers who want to compare grip
force strategies and measure what each one is worth before committing to
hardware.

Everything here is a simulation. No electromyogram is recorded and no hardware is
driven. The file [docs/design-notes.md](docs/design-notes.md) states precisely
which conclusions therefore do not follow from it.

## Problem

A hand prosthesis has one job that a robot arm does not: it has to hold an object
that the controller knows nothing about. The mass, the friction coefficient
against the fingertip, and the stiffness are all unknown, and the only signals
available are the grip force and whatever the tactile sensor reports. Squeeze too
little and the object slides out. Squeeze too much and the paper cup collapses,
the actuator stalls, and the battery runs down.

The user interface makes this harder. A conventional two site myoelectric
interface has two electrodes and therefore one degree of freedom, while an
anthropomorphic hand has ten or more. The user can command one closing velocity,
so the controller has to supply everything else.

Four questions follow, and this library answers each with a number:

1. Given link lengths and a grasp posture from a published taxonomy, what
   opening does the hand present, is the posture inside the joint limits, and
   which objects fit?
2. Given a pair of activation envelopes, what closing velocity should be
   commanded, how is a grasp selected without a third electrode, and how much
   delay does the mapping add?
3. Given contact with an object of unknown stiffness, what commanded force does
   the loop settle on, and how much finger travel does that cost on a rigid
   object against a deformable one?
4. Given that the object starts to slide, how quickly is that detected, how much
   does the object move before the grip force arrests it, and which objects can
   the controller not hold at all?

## Approach

The hand is five serial chains of three segments each, sized from the
anthropometric measurements of Buchholz, Armstrong and Goldstein (1992). Each
finger carries one actuator, and its two distal joints follow the base joint
through fixed ratios, which is how an underactuated prosthetic finger transmits
one motor through a linkage (Birglen et al., 2008; Belter et al., 2013). The
proximal interphalangeal joint follows the metacarpophalangeal joint one for one,
and the distal interphalangeal joint follows the proximal one at two thirds, the
relation used in hand models since Rijpkema and Girard (1991). The thumb has two
degrees of freedom, flexion and opposition, where opposition rotates the thumb
flexion plane from across the palm to palmar so that the thumb pad faces the
finger pads.

Six grasps are implemented as data, taken from the GRASP taxonomy of Feix,
Romero, Schmiedmayer, Dollar and Kragic (2016) and classified on the power to
precision axis of Cutkosky (1989). Each records its taxonomy number, its
opposition type, its thumb position, its open and closed joint postures, and the
digits expected to contact the object. The contact expectation is what turns a
posture into a mechanical prediction, because it fixes how many surfaces share
the tangential load.

Control is the conventional two site arrangement reviewed by Fougner et al.
(2012): the difference of two activation envelopes is passed through a dead zone,
a gain and a saturation to produce a closing velocity. Grasp selection uses
co-contraction, and the recogniser requires the two envelopes to be both active
and balanced, which is what separates a genuine co-contraction from a strong
single site effort. The latency budget follows Farrell and Weir (2007), who found
the controller delay users tolerate to be 100 ms to 125 ms.

Contact is Hertzian, with the fingertip pad and the object treated as two springs
in series and the energy loss added as the velocity proportional term of Hunt and
Crossley (1975). The series stiffness is dominated by the softer body, so the
same commanded opening produces very different forces on a glass and on a foam
cup.

The grip force loop follows the phase structure Romano et al. (2011) use for a
tactile robotic grasp controller: close under the user's command, detect contact,
ramp to a nominal force, then hold and raise the force only when slip is
reported. The nominal force is deliberately light, and slip detection is what
makes a heavy or slippery object holdable, which is the division of labour
Johansson and Westling (1984, 1987) measured in the human precision grip. Slip is
detected from the band limited energy of the tactile signal, following the stress
rate sensing of Howe and Cutkosky (1993).

The alternatives that were considered and rejected, including pattern recognition
control, an integral term in the force loop, and a full rigid body grasp
simulation, are recorded in [docs/design-notes.md](docs/design-notes.md).

## Installation

Requires Python 3.12 or later.

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

## Usage

Run one grasp trial and read off what happened:

```python
from hand_controller import reference_trial, simulate, summarise

trace = simulate(reference_trial("plastic_bottle"))
report = summarise(trace)

print(report.required_force)      # 2.1247741666666666 N
print(report.final_command)       # 2.6 N, after one slip response
print(report.total_slip)          # 0.0036666003345815702 m
print(report.slip_recovery_time)  # 0.073 s
print(report.success)             # True
```

Evaluate the proportional control law and its latency:

```python
from hand_controller import ProportionalConfig, command_latency, control_law

config = ProportionalConfig()
print(control_law(config, 0.05))       # 0.0, inside the dead zone
print(control_law(config, 0.40))       # 0.7333333333333335 closure per second
print(control_law(config, 0.90))       # 1.25, saturated
print(command_latency(config, 0.001))  # 0.047 s against a 0.100 s budget
```

Compare the static contact equilibrium of a rigid object with a deformable one:

```python
from hand_controller import default_hand, equilibrium_force, grasp, opposition_span
from hand_controller.model import default_pad

hand, pad = default_hand(), default_pad()
wrap = grasp("medium_wrap")
print(opposition_span(hand, wrap, 0.0))             # 0.0722197196034674 m
print(opposition_span(hand, wrap, 1.0))             # 0.02772522459903626 m
print(equilibrium_force(0.061, 0.065, pad, 5.0e7))  # 17.233087623506012 N
print(equilibrium_force(0.061, 0.065, pad, 8.0e3))  # 0.6061472451217309 N
```

Runnable examples live in `examples/`:

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
accepts a reduced `--duration` for quick runs. Figures are written to `figures/`,
which is not tracked.

## Results

All numbers below are the output of the commands shown. The configuration is the
reference hand described above, a control period of 1 ms, a trial of 3.0 s with
the object taking up its own weight at 1.0 s, a nominal grip force of 1.20 N, a
safety limit of 15.0 N per contact, and a slip response that doubles the demand
and adds 0.20 N.

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
a two digit pinch carries at the same force.

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
compute per period  4.5 us against a 1000 us period, real time factor 224
```

The characteristic is exactly zero at every sampled point inside the dead zone,
non zero at every point outside it, monotone across the whole range, and flat at
the configured saturation. The 47.0 ms proportional latency is the 90 percent
rise time of the commanded velocity after a step in the closing envelope,
measured end to end through the 50 ms smoothing window and the command slew
limit. It sits inside the 100 ms to 125 ms optimum that Farrell and Weir (2007)
measured. The 178.0 ms mode switch latency is the delay from the onset of a
co-contraction to the grasp changing, against a budget of 250 ms. The throughput
figure is machine dependent and is reported rather than asserted.

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
middle of a reach. Requiring the two envelopes to be within 0.15 of each other
rejects it while still accepting an imbalance of 0.10 in a genuine
co-contraction. The velocity column shows the other half of the arrangement: a
confirmed or refractory co-contraction gates the proportional command to zero, so
selecting a grasp never moves the hand.

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
all.

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
between them.

Force overshoot is 0.00 percent on every object, and the steady state error at the
end of the trial lies between 7.550e-15 N and 2.593e-13 N. Neither is a tuning
achievement. The plant from closure rate to force already contains an integrator,
so a proportional law converges without offset, and the demand is rate limited, so
there is nothing for the loop to overshoot. An integral term would add windup and
buy nothing.

### Grasp success over the object set

`uv run python examples/grasp_evaluation.py`. Ten objects, each with the grasp of
the taxonomy that suits it. `req_N` is the grip force per contact needed to hold
the object against gravity, `cmd_N` is what the controller settled on knowing none
of the object properties, `over_%` is the overshoot of the initial load ramp,
`slip_mm` is the total distance the object slid, `recov_ms` is the time from the
start of sliding to the end of it, and `drop_ms` is the time from the lift to the
object passing the 20 mm drop distance:

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
steel_ball      power_sphere         0.179   0.225  19.221  15.000  15.000    0.00     gone      n/a       66         dropped

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

Nine of the ten objects are held, a success rate of 90.0 percent. Time to contact
ranges from 0.147 s on the apple, which is nearly as wide as the hand opens, to
0.810 s on the pen, which needs almost the whole closing travel of a tripod.
Seven objects start to slide when the load is applied, because the nominal 1.20 N
is below what they need, and six of the seven are recovered. No held object comes
near its crush limit: the water filled paper cup peaks at 2.600 N against a limit
of 4.000 N and the foam cup at 1.200 N against 2.000 N.

### The failure case

The steel ball is a 62 mm solid polished sphere weighing 0.980 kg with a friction
coefficient of 0.10 against the fingertip. From the same script:

```
Failure case: steel_ball
object              steel_ball
grasp               power_sphere (Power Sphere)
required force      19.221 N
commanded force     15.000 N
peak force          15.000 N
slip detections     1
peak slip speed     6636.33 mm/s
drop distance passed 0.066 s after the lift
force saturated     True
outcome             dropped
explanation         holding this object needs 19.22 N per contact, which is 1.28 times the 15.0 N safety limit
```

The controller does everything available to it and still fails. Slip is detected
6 ms after the ball starts to move, the demand is raised on every response until
it reaches the 15.000 N safety limit, and the limit then binds: five contacts at
a friction coefficient of 0.10 and 15.000 N carry 7.5 N of tangential load
against a weight of 9.611 N. The ball passes the 20 mm drop distance 0.066 s
after the lift. This is not a controller defect. The required 19.221 N exceeds
the safety limit by 28 percent, and the limit exists because a hand that can
squeeze harder than that damages what it is holding and overloads its own
actuator. The correct answer for this object is to refuse it, and the trace shows
the refusal in the saturated command rather than in an unbounded force. Holding
it would need either a higher force limit, a higher friction fingertip, or a
grasp with more load bearing contacts.

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
steel_ball              gone          n/a     dropped      dropped           65
-------------------------------------------------------------------------------
objects held               9                                     3
```

Nine objects held with the slip response against three without it. The three that
survive without it are the three that never needed more than the nominal force.
Every other object is on the floor within 94 ms to 168 ms of taking up its own
weight.

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
steel_ball          1.000     1.006            6.0          n/a     gone      6636.33
```

Detection takes 6 ms to 12 ms from the first movement, on the scale of the 6 ms
envelope time constant plus the 4 ms confirmation. Recovery, measured from the
start of sliding to the end of it, takes 28 ms to 96 ms and costs 0.46 mm to
4.58 mm of sliding. The spread follows the size of the shortfall: the drinking
glass needed 1.525 N and had 1.200 N, so it barely accelerated, while the battery
cell needed 2.288 N on two contacts and slid 4.58 mm before the doubled demand
arrived.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/hand_controller/model/anatomy.py` | Link lengths, joint ranges, pad radii, and the underactuated joint coupling |
| `src/hand_controller/model/kinematics.py` | Forward kinematics of every digit, thumb opposition, and joint limit checking |
| `src/hand_controller/model/grasps.py` | The grasp taxonomy as data, the opposition span, and the closure solver |
| `src/hand_controller/model/objects.py` | Object properties and whether a grasp can enclose one |
| `src/hand_controller/model/contact.py` | Hertzian series stiffness, Hunt-Crossley force, contact detection, friction capacity |
| `src/hand_controller/algorithm/protocols.py` | The control law, force regulator and slip detector interfaces the loop depends on |
| `src/hand_controller/algorithm/proportional.py` | Dead zone, gain and saturation, envelope smoothing, and the latency measurement |
| `src/hand_controller/algorithm/modeswitch.py` | Co-contraction recognition, grasp cycling, and the complete two site front end |
| `src/hand_controller/algorithm/force.py` | Grip force regulation with an online plant gain estimate, and the safety clamp |
| `src/hand_controller/algorithm/slip.py` | Band pass, rectify, smooth and threshold slip detection |
| `src/hand_controller/pipeline/emg.py` | Simulated activation envelopes described as trapezoidal bursts |
| `src/hand_controller/pipeline/simulation.py` | The closed loop, the tactile sensor model, and the recorded trace |
| `src/hand_controller/pipeline/scenarios.py` | Named configurations shared by the examples, the tests and the regression file |
| `src/hand_controller/analysis/metrics.py` | Success verdicts, timings, force statistics, and slip episodes |
| `src/hand_controller/analysis/report.py` | Fixed width text rendering of every result table |
| `src/hand_controller/analysis/figures.py` | Control characteristic, span, force profile and slip comparison figures |
| `examples/` | Argument parsing and calls into the library, with no computation of their own |

The layers depend downward only. `model` imports nothing from the package,
`algorithm` imports only its own protocols, `pipeline` imports both, and
`analysis` imports all three. Nothing in `model` or `algorithm` performs input or
output, and no module other than `analysis/figures.py` imports Matplotlib. The
closed loop reaches the force regulator and the slip detector only through the
protocols in `algorithm/protocols.py`, so either can be replaced without touching
the loop.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

There are 2897 tests and the whole suite runs in about thirty seconds.

The property and invariant tier asserts that forward kinematics reproduces the
phalanx lengths and matches a chain summed by hand, that abduction is a pure
rotation of the finger plane, that the joint coupling equals the declared ratios
exactly rather than approximately, that every grasp of the taxonomy is inside its
joint limits and closes monotonically, tested for all six rather than for one,
that the proportional law is zero at every one of 401 sampled points inside the
dead zone and non zero at every point outside it, that it is monotone across 4001
points and flat at the saturation, that a strong single site contraction never
triggers a mode switch at any of 201 activation levels and that a strong
contraction with crosstalk never triggers one at any of 76 crosstalk levels, that
contact detection fires when and only when the indentation is positive across
2001 points spanning the sign change, that the grip force converges to the
commanded force within a hundredth of the controller's convergence band on a
rigid object, that a deformable object reaches a different equilibrium than a
rigid one under the same command and needs more than five times the travel, that
injected slip is detected and the resulting force increase stops it, measured by
running the same object with the response disabled and comparing, and that the
commanded force stays inside the safety interval under a thousand adversarial
inputs including infinities and quiet not a number values.

The regression tier recomputes the whole evaluation set and compares it against
`tests/data/reference_run.json`. What is pinned is deliberate: converged grasp
geometry, root finder results, verdicts, counts, event times and settled forces.
What is not pinned is the raw late run state of a contact simulation, because a
value from an iterative solve that has not converged is not reproducible on
another machine. Every tolerance is derived from the measurement rather than from
an observed error. Timings use one control period, forces use the controller's
convergence band, the closure solver uses a thousand times its own bracket width,
and the slip displacement uses the distance the object covers in one control
period at its peak sliding speed. Run
`uv run python tests/test_regression.py` to regenerate the file after a reviewed
change of behaviour.

The integration tier loads every script in `examples/`, runs it with a reduced
duration, and checks that it exits cleanly and writes output. A separate test
asserts that no example is missing from that table, and four tests exercise the
figure writing paths.

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
| [Matplotlib](https://matplotlib.org/) >= 3.9 | Control characteristic, span, force profile and slip comparison figures | Matplotlib licence, a BSD compatible PSF style licence |
| [pytest](https://pytest.org/) >= 8.3 | Test runner for all three test tiers | MIT |
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
