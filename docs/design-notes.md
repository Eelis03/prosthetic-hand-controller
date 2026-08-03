# Design notes for Prosthetic Hand Controller

## Method selection

### Grasp taxonomy

The grasps are taken from the GRASP taxonomy of Feix, Romero, Schmiedmayer,
Dollar and Kragic (2016), which classifies thirty three human grasps by
opposition type, thumb position, and the power to precision axis of Cutkosky
(1989). Six are implemented: Medium Wrap (3), Prismatic Four Finger (6), Palmar
Pinch (9), Power Sphere (11), Tripod (14) and Lateral (16). The subset was chosen
so that all three opposition types, both thumb positions and all three power
classes appear, and so that it matches the grasp set that commercial
multi-articulating hands provide (Belter et al., 2013).

The taxonomy is data, not code. Each entry carries a taxonomy number, a
classification, an open and a closed joint posture, and the digits expected to
contact the object. The contact expectation is the part that does mechanical
work: it fixes how many surfaces share the tangential load, which is what makes a
six contact wrap hold three times as much as a two digit pinch at the same grip
force.

The assumption this rests on is that the load is shared equally across the
expected contacts, each contributing the coefficient of friction times the grip
force. That is a simplification of grasp wrench analysis, and it is discussed
under limitations below.

Grasps with no opposition, such as the Fixed Hook (15) that carries a bag handle
in the crook of the fingers, were deliberately excluded. They hold a load without
squeezing it, so a grip force controller has nothing to regulate and there is no
span between opposing surfaces to define. Attempting to model one inside the same
framework produced a quantity that was neither the hook opening nor a grip, and
`opposition_span` now raises rather than returning it.

### Hand kinematics

Segment lengths are the male means of Buchholz, Armstrong and Goldstein (1992).
Each finger is a three segment planar chain whose flexion plane is the sagittal
plane rotated about the dorsal axis by an abduction angle. That is enough to
place fingertips and to produce the spread posture of a power sphere, and it
leaves out the small amount of axial rotation the real metacarpophalangeal joint
allows.

The joints are coupled rather than independent, because an underactuated
prosthetic finger drives three joints from one motor through a linkage or a
tendon (Birglen et al., 2008). The proximal interphalangeal joint follows the
metacarpophalangeal joint one for one and the distal interphalangeal joint
follows the proximal one at two thirds, the ratio used in hand models since
Rijpkema and Girard (1991). Ten commanded numbers therefore expand to fifteen
joint angles, and the tests assert that the expansion is exactly the declared
ratio rather than approximately it.

The thumb has two degrees of freedom, flexion and opposition. Opposition rotates
the thumb flexion plane about the thumb's own long axis, from flexing across the
palm at zero to flexing towards the palmar side at ninety degrees. The thumb
angles of each closed posture were selected by sweeping the thumb over both
degrees of freedom and taking the configuration that closes the span
monotonically and leaves the opposing pads in contact. The taxonomy fixes which
surfaces oppose; the geometry of this particular hand fixes the angles that
achieve it.

### Contact

Contact is Hertzian. A curved elastic contact obeys a three halves power law
(Johnson, 1985), and the fingertip pad and the object are treated as two such
springs in series, which inverts in closed form because both carry the same
force. Energy loss during approach is the velocity proportional term of Hunt and
Crossley (1975) rather than a linear dashpot, so the force is continuous at
touchdown instead of stepping.

The consequence that the whole project turns on is that the series stiffness is
dominated by the softer body. Against a rigid glass the effective stiffness is
within four percent of the pad alone; against foam it is close to the object
alone, seventy times smaller. The same commanded opening therefore produces very
different forces, and the same force costs very different amounts of finger
travel. That is a real and testable distinction rather than a modelling detail,
and it is what the force loop has to cope with.

### Proportional myoelectric control

The front end is the conventional two site arrangement reviewed by Fougner et al.
(2012). Two activation envelopes, one from a flexor site and one from an extensor
site, are smoothed, differenced, and mapped through a dead zone, a gain and a
saturation to a closing velocity.

The dead zone is applied with rescaling, so that the surviving activation is
stretched back over the full range. A dead zone without rescaling steps
discontinuously from zero to the dead zone width times the gain at its edge,
which a user feels as a lurch. With rescaling the map is continuous, monotone and
odd, and the tests check all three densely rather than at a handful of points.

Grasp selection uses co-contraction, which is the standard signal because the
differential map ignores it: contracting both sites together leaves the
difference near zero. Recognising it needs three conditions, and the second is
where the engineering is. Both envelopes must exceed an activation threshold,
they must be within a balance tolerance of each other, and the pattern must hold
for a confirmation time. Without the balance condition, an ordinary hard closing
effort with electrode crosstalk satisfies the activation threshold on both
channels and changes grasp in the middle of a reach, which is worse than having
no mode switching at all. The measured result is in the README: a pattern of 0.45
on the opening channel and 0.90 on the closing channel is rejected, while an
imbalance of 0.10 in a genuine co-contraction is accepted.

The proportional command is gated to zero while a co-contraction is being
confirmed and through the refractory interval that follows a switch, so selecting
a grasp never moves the hand.

### Latency budget

Two budgets are stated, and both are measured end to end rather than added up
from parts.

The proportional path is budgeted at 100 ms from a change in the envelope to
ninety percent of the resulting command, which is the lower end of the 100 ms to
125 ms optimum that Farrell and Weir (2007) measured for controller delay in a
myoelectric prosthesis. It is spent on a 50 ms rectangular smoothing window, a
command slew limit that needs 42 ms to sweep the whole command range at the
configured limit, and one 1 ms control period. The slew limit overlaps the window
rather than adding to it, because the smoothed envelope already ramps, which is
exactly why the figure is measured instead of assumed. The measured value is
47.0 ms.

The mode switch is budgeted at 250 ms, because a discrete selection can afford
more delay than a tracking command. It is spent on the same 50 ms window, a
150 ms confirmation hold, and one control period. The measured value is 178.0 ms.

A rectangular moving average was chosen over an exponential one for the envelope
smoothing precisely so that its contribution to the budget is the window length
rather than a settling time that has to be quoted with a tolerance.

### Grip force loop

The phase structure follows Romano et al. (2011): close under the user's command,
detect contact, ramp the demanded force to a nominal value, then hold and raise
the force only when slip is reported. The controller is told nothing about the
object, so the nominal force is deliberately light at 1.20 N and the slip
response is what makes a heavy or slippery object holdable. That is the division
of labour Johansson and Westling (1984, 1987) measured in the human precision
grip, where the grip force is set just above the slip ratio and raised within
about seventy milliseconds of a slip signal from the tactile afferents.

Two decisions inside the loop deserve recording.

The control law is proportional with no integral term. The actuator is commanded
in closure rate and the force depends on closure through a static contact, so the
plant already contains an integrator and a proportional law has no steady state
error. The measured steady state error at the end of a trial is between
7.550e-15 N and 2.593e-13 N. An integral term would add windup during the
approach, when the error is large and the actuator is rate limited, and would buy
nothing.

The proportional gain is divided by an online estimate of how much force one unit
of closure buys. That quantity differs by two orders of magnitude between a glass
and a foam cup, and a fixed gain tuned for one is either unstable or uselessly
slow on the other. The estimate is a bounded, low pass filtered secant computed
from the increments the loop itself produces, and it starts deliberately high so
that the loop is conservative before it has any evidence.

### Slip detection and the response

Slip is detected from the band limited energy of the tactile signal, following
the stress rate sensing of Howe and Cutkosky (1993), who showed that a sensor
responding to the rate of change of contact stress sees the vibration of a
sliding contact long before the static force reveals anything. The chain is a
second order Butterworth band pass over 30 Hz to 300 Hz, rectification, a 6 ms
energy envelope, and a threshold with hysteresis and a 4 ms confirmation.

Choosing the trigger level is the part that decides whether the loop is stable,
and it took a measurement to get right. Every deliberate change of grip force is
itself a transient in the tactile signal, and a band pass does not remove all of
it. Measured across the whole object set, the loop's own force changes leave at
most 0.058 in the envelope with no slip present, while a sliding object produces
hundreds. The first working threshold was set at 0.050, below the artefact, and
the result was a loop that excited itself: raising the force produced a
transient, the transient was read as slip, the response raised the force again,
and the commanded force climbed to the 15 N safety limit on objects that were not
moving at all. The trigger now sits at 0.400, seven times the artefact and three
orders of magnitude below a genuine slip. Raising it costs almost nothing in
detection delay, because the vibration amplitude is proportional to sliding speed
and an object that has begun to slide crosses both levels within a fraction of
the 6 ms envelope time constant. A test replays a real grip force history into a
fresh detector and asserts that it never fires.

The response doubles the demand and adds 0.20 N, clamped to the safety limit. Two
conditions gate it beyond the detector: a 150 ms refractory interval, and a
requirement that the previously demanded force has actually been delivered.
Without the second condition the ladder climbs on transport delay rather than on
evidence, and the force ends far above what the object needed.

## Rejected alternatives

### Pattern recognition control

Pattern recognition control classifies a feature vector extracted from several
electrodes into one of a set of intended motions, which is the approach of
Englehart and Hudgins (2003) and the subject of the review by Scheme and
Englehart (2011).

What it buys: many more classes than the one degree of freedom a two site
interface offers, so a grasp can be selected directly by attempting it rather
than by cycling through a list with co-contractions. That removes the 178 ms
switch latency and, more importantly, removes the need to cycle at all, which on
a six grasp hand can mean five switches to reach the grasp the user wants.

What it costs: several electrodes rather than two, a training session per user
and often per electrode donning, sensitivity to limb position, electrode shift
and skin impedance changes, and a classification decision that is either right or
wrong rather than proportional. Scheme and Englehart (2011) record that the
robustness of these systems outside the laboratory, not their classification
accuracy inside it, is what has limited clinical uptake. A classifier also needs
a window of raw electromyogram, typically 150 ms to 250 ms, which competes
directly with the latency budget.

It was rejected here because this project has no electromyogram at all: the input
is a pair of activation envelopes. Implementing a classifier would have meant
inventing raw signals to classify, which would have tested the invention rather
than the controller. The two site path is also the honest baseline, because it is
what the majority of fitted myoelectric hands use.

### An integral term in the force loop

A proportional integral controller is the reflex choice for a force loop.

What it buys: rejection of a constant disturbance, and zero steady state error
against a plant that has none of its own.

What it costs: windup during the approach, when the force error is large and the
actuator is at its rate limit, and a second gain to tune against a plant whose
gain already varies by two orders of magnitude.

It was rejected because the plant here is an integrator already. The measured
steady state error of the proportional law is at the level of double precision
rounding, so there is nothing left for an integral term to remove.

### Adaptive sliding mode slip prevention

Engeberg and Meek (2013) drive a prosthetic hand with an adaptive sliding mode
controller that prevents slip while minimising deformation of the grasped object.

What it buys: a formal robustness guarantee against an unknown plant, and a
principled trade off between holding and crushing rather than the fixed doubling
rule used here.

What it costs: a sliding surface and adaptation laws to tune, chattering to
suppress, and a structure that is considerably harder to explain than a
proportional law whose gain is normalised by a measured plant gain. It also does
not obviously improve the answer for this object set: the fixed rule already
holds nine of ten objects with peak forces well inside every crush limit, and the
one failure is a force limit rather than a control law.

It was rejected on the ratio of complexity to benefit, not on principle. The slip
response is a small, replaceable part of the loop, and the protocol in
`algorithm/protocols.py` exists so that a different regulator can be dropped in.

### A full rigid body grasp simulation

The alternative to a one dimensional span and a scalar normal force is a rigid
body simulation with per contact wrenches, a friction cone at each contact, and a
grasp matrix.

What it buys: correct treatment of torques about the object, rolling and pivoting
contacts, grasps that fail by rotating rather than by sliding, and a grasp
quality measure that means something.

What it costs: a contact solver, a set of numerical parameters that dominate the
answer, and a large increase in the surface area that has to be tested. The
questions this project asks, how much force the loop settles on and how far the
object slides before the response arrests it, do not need any of it.

It was rejected because the reduced model answers the questions asked and can be
tested exactly. Its cost is recorded under limitations.

### Fixed gain force control

A single proportional gain, tuned once, is the simplest possible force loop.

What it buys: no estimator, no division, and nothing that can adapt in the wrong
direction.

What it costs: on this object set the plant gain from closure to force spans two
orders of magnitude, so a gain that settles a glass in a few milliseconds leaves
a foam cup still moving after a second, and a gain that suits the foam cup is
aggressive enough on the glass to be worth worrying about. The measured travel
figures make the size of the problem concrete: 0.567 mm of indentation on the
glass against 5.280 mm on the paper cup for the same 2.600 N.

It was rejected in favour of dividing the gain by a bounded online estimate of
the same quantity, which costs one filtered division and is tested for staying
inside its bounds under random input.

### An exponential envelope smoother

An exponential moving average is one multiply per sample and needs no buffer.

What it buys: less state, and a smoother frequency response than a rectangular
window.

What it costs: its contribution to the latency budget is a settling time rather
than a length, so the budget can only be stated with a percentage attached. Given
that the latency budget is one of the results this project reports, a filter
whose delay is exactly its window length was worth the fifty element buffer.

## Closed limitations

### An object that has been dropped no longer slides through the fingers

This section used to be a limitation. The model integrated the slide of an object
for the whole trial, whether or not the object was still in the hand, and the
note recorded that a failed trial therefore reported a slide of metres and that
the displacement past the drop distance should not be read as a physical
quantity. The excuse offered for keeping it was that a failed run then showed the
commanded force saturating at the safety limit.

The excuse did not survive being checked. In the failed trial the steel ball
passes the 20 mm drop distance 0.066 s after the lift, and the commanded force
first touches the 15.0 N limit 0.457 s later. The saturation the note was
protecting was produced entirely by a ball that had been on the floor for almost
half a second, and the demand ladder had climbed there by responding to the slip
of an object that was no longer there. The slide the failure case reported,
8.50 m at a peak speed of 6636.33 mm/s, was four hundred times the distance at
which the object is declared lost and a hundred times the length of a finger. It
was not a quantity about a hand.

The trial now ends on the sample the object leaves the hand. `GraspTrace` carries
a `released` flag and is as long as the trial lasted rather than as long as it was
configured for, so every recorded row describes an object that was still between
the fingers.

What it bought. The failure case reports a slide of 20.35 mm at 588.23 mm/s,
both physical, and `test_no_object_slides_further_or_faster_than_it_can` now
holds every trial to the drop distance plus one control period of travel, and to
the free fall speed over the distance it covered. The old model failed the first
of those by four hundred times. The explanation of the failure is now the true
one: the ball needs 19.221 N per contact against a 15.0 N limit and was never
holdable, and it is also gone 0.066 s after the lift, before the 150 ms
refractory interval allows a second slip response, so no ladder could have
reached that force in time. The `slip_mm` column can print a distance for a lost
object instead of the word gone.

What it cost, in four places.

* No trial of the evaluation set saturates the commanded force any more, so the
  set no longer demonstrates the safety clamp. That claim now rests where it
  should have rested all along, on the thousand adversarial inputs in
  `tests/test_force.py`, which include infinities and quiet not a number values.
* `peak_force` and `final_force` of the lost trial moved from pinned to bounded.
  The trial now ends 60 ms into the rise that follows a slip response, which is
  the ill conditioned region described below, and they move by up to 2.71e-2 N
  under a 1e-12 perturbation against 7.0e-14 N for a trial that settles. They are
  bounded by `final_command`, which is pinned and exact. Nothing else moved from
  one group to the other: every other pinned field, both slip bounds, the test
  that switches the slip response off to prove the bounds have teeth, and the
  perturbation tests at 1e-12 and 1e-9 are untouched.
* `steady_state_error` is now only meaningful for a trial that reached a steady
  state. The lost trial stops in the middle of a transient and reports 5.457e-01
  N, which is a fact about when the trial ended and not about the control law.
  The tests say so explicitly rather than averaging it away.
* A trace is no longer guaranteed to have one row per control period, so anything
  reading `trace.config.steps` instead of `len(trace)` is now wrong.

Nine of the ten trials are bit for bit unchanged, because an object that is held
never reaches the drop distance.

## Known limitations

This is a simulation with a simplified contact model. No real electromyogram is
recorded, no classifier is trained, no hardware is driven, and no measurement of
any kind comes from a physical hand. Every number in the README is the output of
the model described here, and the model is only as good as its assumptions.

**The grasp is reduced to one span and one normal force.** There is no grasp
matrix, no friction cone per contact, and no torque about the object. The
tangential load is a single scalar along one axis and the object either slides
along it or does not. Consequences: a grasp that would fail by rotating in the
hand is reported as holding; an object held off centre is indistinguishable from
one held on centre; and the friction capacity is the sum over the expected
contacts of the coefficient of friction times the grip force, which assumes the
load is shared equally and that every contact presses with the same force.
Removing this would mean a contact solver and a rigid body integrator.

**Two site proportional control reaches few degrees of freedom.** One difference
signal commands one velocity. Everything else, which grasp, how much force, and
when to stop, has to come from the controller or from a co-contraction. Selecting
among six grasps by cycling costs up to five co-contractions of 178 ms each, plus
the refractory interval, which is a real burden that this project measures rather
than hides. Pattern recognition control removes it, at the costs recorded above.

**The tactile sensor is a model, not a device.** It reports the normal force plus
a sinusoid whose amplitude is proportional to sliding speed, plus white noise. A
real stress rate sensor produces a broadband, texture dependent signal whose
amplitude depends on the surface as much as on the speed, and the detection
thresholds quoted here would have to be recalibrated against a real one. The
detector structure, a band pass and an energy threshold with hysteresis, would
survive that; the numbers would not.

**Slip is detected after it starts, not before.** The signal used is the
vibration of an already sliding contact. Incipient slip, the partial slip at the
edge of a contact patch before gross sliding begins, is not modelled, so the
object always moves some distance before the response arrives. The measured
distance is 0.46 mm to 4.58 mm depending on the size of the force shortfall.

**The transient of an arrest is not reproducible to the last digit.** The force
regulator estimates its plant gain from a finite difference, a ratio of two small
increments, which is ill conditioned by construction. In the first control
periods after a slip response it turns a relative 1e-12 change in the geometry
into a seven percent change in grip force, and the deceleration of the sliding
object then differs throughout the arrest. The settled force is identical either
way, and every verdict, count and settled quantity is exact, but
``slip_recovery_time`` moves by up to 19 control periods, ``total_slip`` by up to
2.08 mm, ``peak_slip_speed`` by up to 43.8 percent, and any force read before the
transient has died by up to 2.71e-2 N, between one arithmetic ordering and
another. Those are reported as measurements of a particular run and bounded
rather than pinned in the regression suite; the reasoning and the numbers are
recorded in the docstring of `tests/test_regression.py`. Removing this would mean
an analytic derivative of the contact model in place of the secant, which would
also remove the loop's ability to work on an object whose stiffness it does not
know.

**The thumb model conflates opposition with abduction.** Opposition is a single
rotation of the thumb flexion plane, with the radial splay of the metacarpal held
fixed. A real trapeziometacarpal joint has two independent degrees of freedom
about oblique axes. The consequence is that the widest opening this hand presents
between the thumb and the index pads is 49.7 mm, narrower than a human hand
manages, which is why a palmar pinch here is limited to objects below that width.

**Object properties are constant.** Stiffness does not change with indentation
beyond the Hertzian power law, friction does not change with normal force,
sliding speed or dwell time, and nothing is wet, greasy or dusty in a way that
changes during a trial. Real slip is strongly rate dependent and real friction
coefficients drift.

**The success criterion is a threshold, not a task.** A trial succeeds when the
object slides less than 20 mm, stops sliding, stays in contact, and is never
squeezed past its crush limit. That says nothing about whether the object ended
up where the user wanted it, whether the grasp was comfortable, or whether the
hand could have been used for anything else at the same time.

**The object set is small and chosen by hand.** Ten objects with properties
selected to span rigid to deformable and light to heavy is enough to expose the
behaviours this project is about, and it is not a benchmark. A success rate of
90.0 percent is a statement about these ten objects under this disturbance, not a
prediction about anything else.
