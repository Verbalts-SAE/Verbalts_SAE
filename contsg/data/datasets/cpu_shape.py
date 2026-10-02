"""CPU background plus single local-event dataset."""
from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


@Registry.register_dataset("cpu_shape_v1")
class CPUShapeDataModule(BaseDataModule):
    """Load pre-split CPU arrays without additional normalization."""