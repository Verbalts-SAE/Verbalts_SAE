"""Four continuous 32-point quarters; preserve the exported chronological split."""
from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


@Registry.register_dataset("weather_four_stage_128_morph")
class Weather128DataModule(BaseDataModule):
    """Independent weather128 registration (no weather36 artifacts)."""