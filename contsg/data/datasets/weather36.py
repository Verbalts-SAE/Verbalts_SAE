"""Single-channel, continuous 36-point weather morphology dataset."""
from contsg.data.datamodule import BaseDataModule
from contsg.registry import Registry


@Registry.register_dataset("weather_three_stage_36_morph")
class Weather36DataModule(BaseDataModule):
    """Load exported arrays without changing their chronological splits."""