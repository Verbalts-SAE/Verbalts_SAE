import torch

from contsg.models.verbalts import VerbalTSModule


def test_condition_dropout_drops_text_per_sample():
    module = object.__new__(VerbalTSModule)
    torch.nn.Module.__init__(module)
    module.condition_dropout = 1.0
    module.train()
    text = torch.ones(2, 1, 3, 4)
    dropped_text = module._apply_condition_dropout(text)
    assert torch.count_nonzero(dropped_text) == 0


def test_cfg_formula_and_scale_validation():
    conditional = torch.tensor([3.0])
    unconditional = torch.tensor([1.0])
    torch.testing.assert_close(VerbalTSModule.combine_cfg(conditional, unconditional, 2),
                               torch.tensor([5.0]))
    try:
        VerbalTSModule.combine_cfg(conditional, unconditional, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative CFG scale was accepted")