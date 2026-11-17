"""Tests of the estimator, the crop rules, the dump format and the metrics (numpy only, no model needed)."""
import numpy as np
import pytest

from wherebits.core import (from_attention, layer_maps, layer_prior, log2_attention, token_scores, where_bits,
                            wherebits, whether_bits)
from wherebits.crop import best_window, bits_window, bits_window_scores, center_window, window_indicator
from wherebits.data import split_key, split_parity
from wherebits.dump import conditions, load_dump, pack_attention, save_dump
from wherebits.metrics import localization, mc_letter, vqa_accuracy
from wherebits.offline import final_map, reduce_dump

RNG = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def rows(L=3, H=4, T=40, N=16, seed=0):
    """Random full attention rows (softmax over T keys) for the four conditions; visual keys are the first N."""
    rng = np.random.default_rng(seed)
    full = {c: softmax(rng.normal(0, 2, (L, H, T))) for c in ("xt", "0t", "xg", "0g")}
    return full, {c: a[..., :N] for c, a in full.items()}


def test_chain_rule():
    full, vis = rows()
    log2a, mass = from_attention(vis)
    # grouped distribution: N visual outcomes and one outcome for all other tokens
    g = {c: np.concatenate([vis[c], 1 - vis[c].sum(-1, keepdims=True)], -1) for c in ("xt", "xg")}
    kl = (g["xt"] * np.log2(g["xt"] / g["xg"])).sum(-1)
    parts = whether_bits(mass["xt"], mass["xg"]) + where_bits(log2a["xt"], mass["xt"], log2a["xg"], mass["xg"])
    np.testing.assert_allclose(kl, parts, rtol=1e-9, atol=1e-12)


def test_shared_logit_term_does_not_change_token_ranking():
    rng = np.random.default_rng(1)
    T, N = 30, 12
    s_t, s_g, b = rng.normal(size=T), rng.normal(size=T), np.zeros(T)
    b[:N] = rng.normal(0, 5, N)                            # e.g. a sink: the same key term under both prompts
    d = []
    for shift in (0.0, 1.0):
        a_t, a_g = softmax(s_t + shift * b)[:N], softmax(s_g + shift * b)[:N]
        d.append(token_scores(log2_attention(a_t), a_t.sum(), log2_attention(a_g), a_g.sum()))
    np.testing.assert_allclose(d[0] - d[0][0], d[1] - d[1][0], atol=1e-9)


def test_layer_maps_and_prior():
    d = RNG.normal(size=(2, 3, 5))
    w = np.array([[1.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    F = layer_maps(d, w)
    np.testing.assert_allclose(F[0], (d[0, 0] + 3 * d[0, 2]) / 4)
    np.testing.assert_allclose(F[1], 0.0)
    pi = layer_prior([1.0, 2.0, 0.0, -1.0], gamma=4)
    np.testing.assert_allclose(pi, [1 / 17, 16 / 17, 0, 0])
    np.testing.assert_allclose(layer_prior([0.0, 0.0]), [0.5, 0.5])


def test_wherebits_variants():
    _, vis = rows(L=4, H=2, N=9)
    log2a, mass = from_attention(vis)
    prior = layer_prior(np.arange(4.0))
    r = wherebits(log2a, mass, prior)
    assert r.map.shape == (9,) and r.layer_maps.shape == (4, 9) and r.head_weights.shape == (4, 2)
    assert (r.head_weights >= 0).all()
    np.testing.assert_allclose(r.map, (prior[:, None] * r.layer_maps).sum(0))
    two = wherebits({c: log2a[c] for c in ("xt", "xg")}, {c: mass[c] for c in ("xt", "xg")}, prior, weighting="mass")
    assert two.map.shape == (9,)
    per_q = wherebits(log2a, mass, None)
    assert np.isclose(per_q.layer_weights.sum(), 1.0) and (per_q.layer_weights > 0).sum() <= 3
    with pytest.raises(ValueError):
        wherebits(log2a, mass, np.ones(3) / 3)


def brute_best_window(m, frac):
    gh, gw = m.shape
    wh, ww = max(1, round(frac * gh)), max(1, round(frac * gw))
    pos = np.clip(m, 0, None)
    best, arg = -1, None
    for i in range(gh - wh + 1):
        for j in range(gw - ww + 1):
            s = pos[i:i + wh, j:j + ww].sum()
            if s > best:
                best, arg = s, (i, j, wh, ww)
    return arg


@pytest.mark.parametrize("frac", [0.5, 0.33])
def test_best_window(frac):
    for seed in range(5):
        m = np.random.default_rng(seed).normal(size=(14, 14))
        assert best_window(m, frac) == brute_best_window(m, frac)
    assert best_window(-np.ones((8, 8)), 0.5) == center_window((8, 8), 0.5)
    win = (3, 2, 7, 7)
    assert best_window(window_indicator((14, 14), win), 0.5) == win


def test_bits_window_matches_definition():
    _, vis = rows(L=2, H=3, N=36, seed=3)
    log2a, mass = from_attention(vis)
    prior = np.array([0.3, 0.7])
    grid, frac = (6, 6), 0.5
    scores, (wh, ww) = bits_window_scores(log2a, mass, prior, grid, frac)
    P = {c: vis[c] / vis[c].sum(-1, keepdims=True) for c in vis}
    w = np.maximum(mass["xt"] * (P["xt"] * np.log2(P["xt"] / P["xg"])).sum(-1)
                   - mass["0t"] * (P["0t"] * np.log2(P["0t"] / P["0g"])).sum(-1), 0)
    i, j = 1, 2
    inside = np.zeros(grid, bool)
    inside[i:i + wh, j:j + ww] = True
    mu, nu = P["xt"][..., inside.ravel()].sum(-1), P["xg"][..., inside.ravel()].sum(-1)
    b = np.where(mu > nu, mu * np.log2(mu / nu) + (1 - mu) * np.log2((1 - mu) / (1 - nu)), 0)
    expected = (prior * (w * b).sum(1) / w.sum(1)).sum()
    assert np.isclose(scores[i, j], expected, rtol=1e-6)
    assert bits_window(log2a, mass, prior, grid, frac)[2:] == (wh, ww)


def test_dump_roundtrip_and_offline(tmp_path):
    _, vis = rows(L=3, H=4, N=16, seed=5)
    arrays = pack_attention(vis)
    arrays["gt"] = np.zeros((4, 4), bool)
    arrays["gt"][1:3, 1:3] = True
    meta = {"qid": "q1", "grid": [4, 4], "image_id": 7}
    save_dump(tmp_path / "q1.npz", arrays, meta)
    d = load_dump(tmp_path / "q1.npz")
    assert d["meta"]["qid"] == "q1"
    log2a, mass = conditions(d)
    np.testing.assert_allclose(log2a["xt"], log2_attention(vis["xt"]), atol=1 / 16 + 1e-6)   # 1/8-bit codes
    rec = reduce_dump(tmp_path / "q1.npz")
    prior = layer_prior(np.array([1.0, 2.0, 3.0]))
    online = wherebits(log2a, mass, prior)
    np.testing.assert_allclose(final_map(rec, prior), online.map, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rec["W"], online.layer_bits, rtol=1e-5, atol=1e-6)
    exact = wherebits(*from_attention(vis), prior)
    assert np.corrcoef(exact.map, online.map)[0, 1] > 0.99            # quantization barely changes the map


def test_localization_metrics():
    g = np.array([0, 1, 1, 0, 0], bool)
    m = np.array([0.1, 0.9, 0.3, 0.5, 0.0])
    out = localization(m, g)
    pairs = [(a, b) for a in m[g] for b in m[~g]]
    assert out["pointing"] == 1.0
    assert np.isclose(out["auroc"], np.mean([a > b for a, b in pairs]))
    assert np.isnan(localization(m, np.zeros(5, bool))["auroc"])


def test_answers_and_split():
    assert vqa_accuracy("Two", ["2", "2", "2", "two", "3", "2", "2", "2", "2", "2"]) == 1.0
    assert mc_letter("(B) red") == "B" and mc_letter("The answer is C.") == "C"
    a = split_key({"image_id": "COCO_train2014_000000296635_2.jpg", "qid": "x"})
    b = split_key({"image_id": "COCO_train2014_000000296635_7.jpg", "qid": "y"})
    assert a == b                                                     # expressions of one image share the split
    cal, ev = split_parity([{"image_id": 3}, {"image_id": 4}])
    assert cal.tolist() == [True, False] and ev.tolist() == [False, True]
