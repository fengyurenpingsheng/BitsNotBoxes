# Reproducing the paper

All commands run from the repository root after the installation in the [README](../README.md). The paper's runs
used CPU nodes (16 cores, float32); every model script accepts `--device cuda` and shards with `--shard/--nshards`
or a SLURM array (`slurm/array.sbatch`). Model lists used below:

```bash
M6="smolvlm-256m smolvlm-500m internvl3-1b qwen3-vl-2b qwen3-vl-8b llava-1.5-7b"   # COCO, RefCOCO
M5="smolvlm-500m internvl3-1b qwen3-vl-2b qwen3-vl-8b llava-1.5-7b"                # V*
M4="smolvlm-500m internvl3-1b qwen3-vl-2b llava-1.5-7b"                             # TextVQA zoom-in
MT="janus-pro-1b janus-pro-7b qwen3-vl-2b-896 qwen3-vl-8b-896 qwen3-vl-2b-native qwen3-vl-8b-native"   # transfer
```

## 1. Data

```bash
python scripts/prepare_coco.py    --data data/coco --out data/coco/questions.jsonl   # COCO val2017, 500 images
python scripts/prepare_refcoco.py --out data/refcoco                                 # 300 testA + 300 testB
python scripts/prepare_textvqa.py --out data/textvqa                                 # 500 validation questions
python scripts/prepare_vstar.py   --out data/vstar                                   # 191 questions
python scripts/download_models.py --models $M6 janus-pro-1b janus-pro-7b              # optional: cache weights
```

`data/paper_questions/*.tsv` lists the question ids of the paper and their split. Localization uses the COCO
presence and color questions: 385 on images with odd ids (calibration split) and 393 on images with even ids
(evaluation). RefCOCO is split by image in the same way (315 / 285).

## 2. Attention dumps

One dump per question and model: the greedy answer, the answer-start attention of every layer and head for the four
input conditions, and the baseline maps of vlmprobe (Chefer relevance, Grad-CAM, rollout, token activation map).

```bash
for m in $M6; do
  python scripts/dump_attention.py --task coco    --model $m --questions data/coco/questions.jsonl    --out results/dumps
  python scripts/dump_attention.py --task refcoco --model $m --questions data/refcoco/questions.jsonl --out results/dumps
done
for m in $M5; do python scripts/dump_attention.py --task vstar   --model $m --questions data/vstar/questions.jsonl   --out results/dumps; done
for m in $M4; do python scripts/dump_attention.py --task textvqa --model $m --questions data/textvqa/questions.jsonl --out results/dumps; done
for m in $MT; do
  python scripts/dump_attention.py --task coco  --model $m --questions data/coco/questions.jsonl  --out results/dumps
  python scripts/dump_attention.py --task vstar --model $m --questions data/vstar/questions.jsonl --out results/dumps
done
```

On 16 CPU cores a dump takes about 7 s per question for SmolVLM-500M, 15-20 s for InternVL3-1B and Qwen3-VL-2B,
35-40 s for Qwen3-VL-8B and 60-75 s for LLaVA-1.5-7B. The four attention passes take about a third of this; the rest
is the answer generation and the baselines (`--no-baselines` and `--no-hidden` skip the baselines and the hidden-state
norms). Dumps of the sensitivity study:
`--generic-prompt describe|see|caption --tag _p-describe` and `--gray 64|192 --tag _g64`.

## 3. Layer prior

The shipped priors (`wherebits/priors/`) are the ones of the paper. To recompute them from the COCO calibration
questions (no labels are read):

```bash
python scripts/calibrate_prior.py --dumps results/dumps/coco --models $M6 $MT --out results/priors
```

For the six main models, recomputing from our dumps gives the shipped priors bit for bit. Pass
`--priors results/priors` to the scripts below to use your own.

## 4. Localization (Tables 1, 4 and 5)

```bash
python scripts/evaluate_localization.py --task coco    --dumps results/dumps --models $M6 --out results/localization
python scripts/evaluate_localization.py --task refcoco --dumps results/dumps --models $M6 --out results/localization
python scripts/evaluate_localization.py --task vstar   --dumps results/dumps --models $M5 --out results/localization
python scripts/evaluate_localization.py --task coco    --dumps results/dumps --models $MT --out results/transfer
python scripts/evaluate_localization.py --task coco    --dumps results/dumps --models $M6 --out results/ablation --ablations
```

Expected means (Table 1; pointing / token AUROC). Our dumps give exactly these numbers:

| Method | COCO (6 VLMs) | RefCOCO (6 VLMs) | V\* (5 VLMs) |
|---|---|---|---|
| ViCrop, tuned layer (labels) | 0.731 / 0.815 | 0.657 / 0.746 | 0.243 / 0.754 |
| **WhereBits** | **0.723 / 0.843** | **0.674 / 0.755** | 0.210 / 0.789 |
| ViCrop, all layers | 0.664 / 0.788 | 0.593 / 0.710 | 0.204 / 0.739 |
| ViCrop, layer band | 0.664 / 0.793 | 0.626 / 0.718 | 0.196 / 0.739 |
| CARVE (re-impl.) | 0.505 / 0.722 | 0.462 / 0.630 | 0.186 / 0.760 |
| Chefer relevance | 0.302 / 0.679 | 0.211 / 0.576 | 0.027 / 0.732 |
| Mean attention | 0.221 / 0.670 | 0.184 / 0.573 | 0.002 / 0.715 |
| KL-weighted map | 0.367 / 0.713 | 0.323 / 0.610 | 0.128 / 0.757 |
| Random | 0.196 / 0.500 | 0.234 / 0.502 | 0.013 / 0.494 |

Transfer (Table 4, COCO, WhereBits): Janus-Pro-1B 0.835 / 0.849, Janus-Pro-7B 0.840 / 0.855, Qwen3-VL-2B at 896 px
0.740 / 0.889 and native 0.786 / 0.891, Qwen3-VL-8B at 896 px 0.880 / 0.908 and native 0.896 / 0.912. ViCrop's
layer is re-tuned with labels for every model and resolution. With `--ablations`, the single-change variants of
Table 5 (uniform layers, prior exponent 1, no gray-image correction in the prior or in the head weights, KL-only,
mass-only and uniform head weights, log-ratio of the head-mean attention) are evaluated on the same questions.

## 5. Zoom-in (Tables 2 and 3)

```bash
python scripts/build_maps.py --task vstar   --dumps results/dumps --models $M5 --out results/maps
python scripts/build_maps.py --task textvqa --dumps results/dumps --models $M4 --out results/maps
for m in $M5; do
  python scripts/zoom_in.py --task vstar --model $m --questions data/vstar/questions.jsonl \
      --maps results/maps/vstar/$m.npz --out results/zoom
done
for m in $M4; do
  python scripts/zoom_in.py --task textvqa --model $m --questions data/textvqa/questions.jsonl \
      --maps results/maps/textvqa/$m.npz --out results/zoom
done
python scripts/summarize_zoom.py --results results/zoom/vstar   --models $M5
python scripts/summarize_zoom.py --results results/zoom/textvqa --models $M4
```

Each map's best window of about 1/2 and 1/3 of the image side is cropped from the original image; the model answers
from the image and the crop (LLaVA-1.5: the crop alone). V\* is scored by forced choice, TextVQA by VQA accuracy of
greedy answers (at most 10 tokens). The maps are stored as float16 bundles, as for the paper; on our dumps the
bundles give the paper's crop for every question, model, window and method of V\* and TextVQA.

Expected accuracy (%, mean of the two windows): V\* at 448 px (5 VLMs) no zoom 44.2, WhereBits 50.2, WhereBits with
the bits window 51.4, ViCrop tuned layer 50.4, ViCrop all layers 48.1, CARVE 48.7, Chefer relevance 46.6, mean
attention 44.7. TextVQA (4 VLMs): no zoom 56.4, WhereBits 62.3, bits window 63.4, CARVE 64.1, Chefer relevance 63.9,
mean attention 63.2, ViCrop all layers 62.3, ViCrop tuned layer 61.7.

## 6. Faithfulness (Fig. 4)

```bash
for m in smolvlm-256m smolvlm-500m internvl3-1b qwen3-vl-2b; do        # 393 evaluation questions, gray and blur
  python scripts/build_maps.py --task coco --dumps results/dumps --models $m --out results/maps
  python scripts/faithfulness.py --model $m --questions data/coco/questions.jsonl \
      --maps results/maps/coco/$m.npz --out results/faithfulness
done
for m in qwen3-vl-8b llava-1.5-7b; do                                     # 200 evaluation questions, gray only
  python scripts/build_maps.py --task coco --dumps results/dumps --models $m --out results/maps --max-questions 200
  python scripts/faithfulness.py --model $m --questions data/coco/questions.jsonl \
      --maps results/maps/coco/$m.npz --out results/faithfulness --operators gray
done
python scripts/summarize_faithfulness.py --results results/faithfulness --models $M6
```

This gives the ten model-operator settings per axis of Fig. 4 (20 in total); `--max-questions 200` draws the same
200 questions as in the paper.

Cells are replaced by gray or blurred pixels in map order (8 budget points); each curve is scored on the token axis
and on the bit axis (CompressAI `mbt2018-mean`, quality 4). Texture-first (`codec_bits`) and flat-first
(`flat_first`) controls are added to every question. The summary reports insertion AUC minus deletion AUC, mean ranks
and Wilcoxon tests against WhereBits with Holm correction over the settings.
