"""Built-in dataset registrations discovered by :class:`contsg.registry.Registry`."""

from contsg.data.datasets.synth_u import SynthUDataModule
from contsg.data.datasets.weather36 import Weather36DataModule
from contsg.data.datasets.weather128 import Weather128DataModule
from contsg.data.datasets.semisynth_morph import ElectricitySemiSynthMorphDataModule
from contsg.data.datasets.etth2_2class import Etth2TwoClassDataModule

__all__ = [
    "Weather128DataModule",
    "Weather36DataModule",
    "SynthUDataModule",
    "ElectricitySemiSynthMorphDataModule",
    "Etth2TwoClassDataModule",
]

# These datasets are optional in lightweight checkouts.  Register them whenever
# their source modules are present without making every other dataset unusable
# when they are absent.
try:
    from contsg.data.datasets.semisynth_morph_v2 import SemiSynthMorphV2DataModule
except ModuleNotFoundError as error:
    if error.name != "contsg.data.datasets.semisynth_morph_v2":
        raise
else:
    __all__.append("SemiSynthMorphV2DataModule")

try:
    from contsg.data.datasets.semisynth_morph_v3 import SemiSynthMorphV3DataModule
except ModuleNotFoundError as error:
    if error.name != "contsg.data.datasets.semisynth_morph_v3":
        raise
else:
    __all__.append("SemiSynthMorphV3DataModule")
