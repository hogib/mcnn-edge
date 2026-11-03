import json

import numpy as np
import pytest

from sonar_edge.cv import fold_gt, load_pred, pool
from sonar_edge.metrics import evaluate


def _fold(rng, imgs, n_classes):
    gt = {i: [(int(rng.integers(n_classes)), 0.0, 0.0, 10.0, 10.0) for _ in range(rng.integers(0, 3))] for i in imgs}
    pred = {i: [(c, float(rng.normal()), 1.0, 1.0, 10.0, 10.0) for c, *_ in gt[i]] + [(0, -1.0, 50, 50, 60, 60)]
            for i in imgs}
    return pred, gt


def test_pool_equals_evaluate_on_union():
    rng = np.random.default_rng(0)
    (p0, g0), (p1, g1) = _fold(rng, range(0, 20), 3), _fold(rng, range(20, 45), 3)
    aps, m = pool([p0, p1], [g0, g1], 3)
    aps_u, m_u = evaluate({**p0, **p1}, {**g0, **g1}, 3)
    assert np.allclose(np.nan_to_num(aps), np.nan_to_num(aps_u)) and m == m_u


def test_pool_rejects_overlapping_folds():
    rng = np.random.default_rng(1)
    p, g = _fold(rng, range(5), 2)
    with pytest.raises(AssertionError):
        pool([p, p], [g, g], 2)


def test_fold_gt_matches_dataset(tmp_path):
    pytest.importorskip("torch")
    from sonar_edge.data import DATA_ROOT
    if not DATA_ROOT.exists():
        pytest.skip("dataset not fetched")
    from sonar_edge.data import class_names, load_frames
    from sonar_edge.dataset import dev_datasets
    frames = load_frames()
    classes = class_names(frames)
    ds, _ = dev_datasets(8, 2, 3, 4)
    assert fold_gt(frames, classes, 3, 2) == ds["test"].gt_boxes()


def test_load_pred_roundtrip(tmp_path):
    p = tmp_path / "pred.json"
    p.write_text(json.dumps({"7": [[1, 0.5, 1, 2, 3, 4]]}))
    assert load_pred(p) == {7: [(1, 0.5, 1, 2, 3, 4)]}
