"""Control laws, mode switching, force regulation and slip detection.

Nothing in this layer imports the pipeline or the analysis, performs input or
output, or draws anything. Everything it needs from the physics arrives as a
number in a function argument.
"""

from hand_controller.algorithm.force import (
    ForceConfig,
    ProportionalForceRegulator,
    SlipResponseConfig,
)
from hand_controller.algorithm.modeswitch import (
    CoContractionConfig,
    ModeSwitcher,
    ModeSwitchResult,
    MyoelectricController,
    MyoelectricOutput,
    SwitchState,
    is_co_contraction,
    mode_switch_latency,
)
from hand_controller.algorithm.proportional import (
    EnvelopeFilter,
    ProportionalConfig,
    ProportionalControlLaw,
    command_latency,
    control_law,
    saturation_difference,
)
from hand_controller.algorithm.protocols import (
    ControlLaw,
    GripForceRegulator,
    SlipDetector,
    SlipReading,
)
from hand_controller.algorithm.slip import BandPassSlipDetector, SlipDetectorConfig

__all__ = [
    "BandPassSlipDetector",
    "CoContractionConfig",
    "ControlLaw",
    "EnvelopeFilter",
    "ForceConfig",
    "GripForceRegulator",
    "ModeSwitchResult",
    "ModeSwitcher",
    "MyoelectricController",
    "MyoelectricOutput",
    "ProportionalConfig",
    "ProportionalControlLaw",
    "ProportionalForceRegulator",
    "SlipDetector",
    "SlipDetectorConfig",
    "SlipReading",
    "SlipResponseConfig",
    "SwitchState",
    "command_latency",
    "control_law",
    "is_co_contraction",
    "mode_switch_latency",
    "saturation_difference",
]
