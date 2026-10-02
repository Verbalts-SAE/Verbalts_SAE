"""Data module registration for the ETTh2 single/double-peak 2-class dataset."""

from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


@Registry.register_dataset(
    "etth2-2class",
    aliases=["etth2_2class"],
)
class Etth2TwoClassDataModule(BaseDataModule):
    """ETTh2 windows with per-segment single-peak / double-peaks labels."""
