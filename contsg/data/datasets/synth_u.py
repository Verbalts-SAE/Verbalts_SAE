"""Data module registration for the Synth-U benchmark dataset."""

from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


@Registry.register_dataset("synth-u")
class SynthUDataModule(BaseDataModule):
    """Synth-U dataset (parameterized morphology on smooth trends)."""
