"""Evaluation metrics: localization (pointing, token AUROC), VQA accuracy and multiple-choice letters."""
from __future__ import annotations

import re

import numpy as np
from scipy.stats import rankdata


def localization(map_, gt) -> dict:
    """Pointing (the map's peak lies on a target token), token AUROC and AP, and the positive map mass on targets.

    AUROC and AP are NaN unless the image has both target and non-target tokens.
    """
    m = np.nan_to_num(np.asarray(map_, np.float64).ravel(), nan=-1e9, posinf=1e9, neginf=-1e9)
    g = np.asarray(gt, bool).ravel()
    n, npos = g.size, int(g.sum())
    out = {"pointing": float(g[int(np.argmax(m))])}
    pos = np.clip(m, 0, None)
    s = pos.sum()
    out["mass"] = float(pos[g].sum() / s) if s > 0 else np.nan
    if 0 < npos < n:
        ranks = rankdata(m)                                   # average ranks for ties
        out["auroc"] = float((ranks[g].sum() - npos * (npos + 1) / 2) / (npos * (n - npos)))
        order = np.argsort(-m, kind="stable")
        hits = g[order]
        prec = np.cumsum(hits) / np.arange(1, n + 1)
        out["ap"] = float((prec * hits).sum() / npos)
    else:
        out["auroc"] = out["ap"] = np.nan
    return out


# ----------------------------------------------------------------------------- VQA accuracy (TextVQA)
_CONTRACTIONS = {"dont": "don't", "doesnt": "doesn't", "isnt": "isn't", "arent": "aren't", "cant": "can't",
                 "wont": "won't", "im": "i'm", "ive": "i've", "youre": "you're", "thats": "that's",
                 "whats": "what's", "its": "it's"}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!*#%^&$:]")
_PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")


def normalize_answer(ans: str) -> str:
    """VQA answer normalization (compact re-implementation of the official EvalAI script)."""
    s = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _PERIOD.sub("", s)
    words = []
    for w in s.split():
        w = _NUMBERS.get(w, w)
        if w in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def vqa_accuracy(pred: str, answers: list) -> float:
    """Official VQA accuracy: mean over the 10 leave-one-out subsets of min(1, #matches / 3)."""
    p = normalize_answer(pred)
    gts = [normalize_answer(a) for a in answers]
    if not gts:
        return float("nan")
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3.0))
    return sum(accs) / len(accs)


def mc_letter(pred: str) -> str:
    """First option letter of a generated multiple-choice answer ("B", "(B)", "B. red", "The answer is B")."""
    s = pred.strip()
    m = re.match(r"^\(?([A-H])\)?(?:[\s.):,]|$)", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-H])\b", s)
    return m.group(1) if m else ""
