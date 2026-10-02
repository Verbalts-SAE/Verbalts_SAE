import numpy as np
import pytest
from scripts.prepare_weather128 import backgrounds, intervals, shape_bank, local_caption


def test_four_stage_shapes_and_captions():
    labels, residuals = shape_bank(1.2)
    assert labels.shape == (81, 4) and residuals.shape == (81, 128)
    assert len(np.unique(labels, axis=0)) == 81
    assert np.all(residuals[:, [0,31,32,63,64,95,96,127]] == 0)
    text = local_caption([1,2,3,1])
    assert text.count('quarter') == 4 and 'fourth quarter has a single peak' in text
    with pytest.raises(ValueError):
        local_caption([1,2,3])


def test_continuous_blocks_and_embargo():
    n = 128*30
    values = np.column_stack([np.arange(n), np.ones(n), np.ones(n)*1000])
    times = np.datetime64('2020-01-01') + np.arange(n)*np.timedelta64(10, 'm')
    split = intervals(n)
    assert split['valid'][0] - split['train'][1] == 128
    bg, records = backgrounds(values, times, split['train'], 6, 42)
    occupied = set()
    for row, record in zip(bg, records):
        start, stop = record['source_start'], record['source_stop']
        assert stop-start == 128 and not occupied.intersection(range(start, stop))
        occupied.update(range(start, stop))
        # Reconstruct from provenance rather than allowing resampling/stitching.
        from scripts.prepare_weather128 import CHANNELS
        raw = values[start:stop, CHANNELS.index(record['channel'])]
        np.testing.assert_allclose(row, (raw-raw[0])*record['scale'])