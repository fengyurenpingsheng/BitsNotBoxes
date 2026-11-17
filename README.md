# Bits, Not Boxes: Label-Free Layer Selection for Visual Localization in VLMs

Haisheng Fu<sup>1</sup>, Gexin Huang<sup>1</sup>, Haikai Zhao<sup>2</sup>, Jie Liang<sup>2</sup>, Zhenman Fang<sup>3</sup>, Jingning Han<sup>4</sup>, Xiaoxiao Li<sup>1</sup>

<sup>1</sup>University of British Columbia &nbsp; <sup>2</sup>Simon Fraser University &nbsp; <sup>3</sup>University of Minnesota–Twin Cities &nbsp; <sup>4</sup>Google

**Paper:** the arXiv link will be added here.

This repository contains **WhereBits**, a training-free method that reads *where a vision-language model (VLM) looks* from its own attention, and the code that produces the results of the paper.

## In short

Attention maps of a VLM are cheap to read, but attention sinks make them unreliable for localization. Contrasting the question with a generic prompt suppresses the sinks, yet only some layers localize the queried object, and existing methods find these layers with box annotations or fixed rules. WhereBits finds them from the attention itself:

1. **Per-head token scores.** For every head, compare the attention of the answer-start token under the question *t* and under a generic prompt *g* ("Write a general description of the image."):
   `d(n) = log2 P_t(n) − log2 P_g(n)`, where `P` is the head's attention distribution over the visual tokens. Any logit term shared by both prompts, such as a sink, cancels in the token ranking.
2. **Where-to-look bits.** The divergence between the two attention rows splits exactly into *whether* the head looks at the image and *where* it looks. The second part, `m_t · KL(P_t ‖ P_g)`, minus the same quantity on a gray image, is the head weight.
3. **Label-free layer prior.** The mean of these bits per layer over **unlabeled** questions gives the layer prior `π_l ∝ B̄_l^4`, computed once per model and input size. The map is `M(n) = Σ_l π_l F_l(n)`, where `F_l` averages the head scores of layer *l* with the head weights.

Across eight VLMs, WhereBits matches the pointing accuracy of box-tuned layers on COCO and RefCOCO and ranks target tokens better (COCO pointing 0.72, against 0.22 for mean attention). In training-free zoom-in, its crops raise V\* accuracy by 6.0 points (forced-choice evaluation).

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/fengyurenpingsheng/BitsNotBoxes.git
cd BitsNotBoxes
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # choose the build for your hardware at https://pytorch.org
pip install -r requirements.txt        # transformers 5.3, vlmprobe, CompressAI, ...
pip install -e .
pip install pytest && python -m pytest -q   # unit tests of the estimator (no model needed)
```

The estimator, the crop rules and the analysis of saved attention dumps need only numpy and scipy (`pip install -e .`). Running a model needs the full `requirements.txt`. Attention is read with [vlmprobe](https://github.com/gexinh/vlm-probe), pinned to a fixed commit.

## Quick start

```bash
python scripts/demo.py --model qwen3-vl-2b \
    --image http://images.cocodataset.org/val2017/000000039769.jpg \
    --question "What color is the remote control?" --device cuda --out demo.png
```

The demo saves the map with the zoom-in window (`demo.png`), the crop (`demo_crop.png`), and prints the model's answers without and with the crop. Use `--device cpu` without a GPU (the paper's experiments ran on CPUs in float32), `--passes 2` for the two-pass variant, and `--rule bits` for the bits-window crop.

## Python API

```python
from PIL import Image
from wherebits import WhereBitsVLM

wb = WhereBitsVLM("qwen3-vl-2b", device="cuda")      # loads the model and its shipped layer prior
out = wb(Image.open("street.jpg"), "What is the color of the umbrella?")

out.map                          # [14, 14] map over the visual-token grid (positive = question attracts attention)
out.result.layer_weights         # layer prior pi_l
out.result.head_weights          # head weights w_lh of this question
crop, box = out.crop(0.5)                 # primary rule: window of half the image side with the largest map sum
crop, box = out.crop(0.33, rule="bits")   # bits-window rule
```

The estimator also works on attention you extracted yourself (numpy arrays `[layers, heads, visual tokens]` of the answer-start row under the four conditions: image or gray image × question or generic prompt):

```python
from wherebits import from_attention, load_prior, wherebits

log2a, mass = from_attention({"xt": A_xt, "xg": A_xg, "0t": A_0t, "0g": A_0g})
res = wherebits(log2a, mass, prior=load_prior("qwen3-vl-2b"))
res.map, res.layer_maps, res.head_weights, res.layer_bits
```

## Supported models

Every model sees one image frame, so the visual-token grid is fixed. A layer prior is shipped for each entry (`wherebits/priors/`).

| Key | Hugging Face model | Frame | Visual tokens | Layers × heads |
|---|---|---|---|---|
| `smolvlm-256m` | HuggingFaceTB/SmolVLM-256M-Instruct | 512 px | 8 × 8 | 30 × 9 |
| `smolvlm-500m` | HuggingFaceTB/SmolVLM-500M-Instruct | 512 px | 8 × 8 | 32 × 15 |
| `internvl3-1b` | OpenGVLab/InternVL3-1B-hf | 448 px | 16 × 16 | 24 × 14 |
| `qwen3-vl-2b` | Qwen/Qwen3-VL-2B-Instruct | 448 px | 14 × 14 | 28 × 16 |
| `qwen3-vl-8b` | Qwen/Qwen3-VL-8B-Instruct | 448 px | 14 × 14 | 36 × 32 |
| `llava-1.5-7b` | llava-hf/llava-1.5-7b-hf | 336 px | 24 × 24 | 32 × 32 |
| `janus-pro-1b` | deepseek-community/Janus-Pro-1B | 384 px | 24 × 24 | 24 × 16 |
| `janus-pro-7b` | deepseek-community/Janus-Pro-7B | 384 px | 24 × 24 | 30 × 32 |
| `qwen3-vl-2b-896`, `qwen3-vl-8b-896` | Qwen3-VL-2B / 8B | 896 px | 28 × 28 | as above |
| `qwen3-vl-2b-native`, `qwen3-vl-8b-native` | Qwen3-VL-2B / 8B | native aspect ratio | ≤ 1280 | as above |

**A new model or input size** needs only unlabeled questions: add a `ModelSpec` to `wherebits/models.py` (frame size and token grid; the model must be supported by vlmprobe), dump the attention of a few hundred unlabeled questions with `scripts/dump_attention.py`, and run `scripts/calibrate_prior.py` (`--all-questions` for your own question set). In the paper, ten unlabeled questions suffice for the layer scores.

## Reproducing the paper

The experiments run in two stages. First, a model pass stores the answer-start attention of every layer and head under the four input conditions for each question (`scripts/dump_attention.py`). Everything else is computed from these dumps without the model, except zoom-in and faithfulness, which run the model again on crops and perturbed images.

| Step | Script | Paper |
|---|---|---|
| Data | `prepare_coco.py`, `prepare_refcoco.py`, `prepare_textvqa.py`, `prepare_vstar.py` | Sec. 4 |
| Attention dumps | `dump_attention.py` | all |
| Layer prior | `calibrate_prior.py` (or the shipped priors) | Sec. 3.4 |
| Localization and ablations | `evaluate_localization.py [--ablations]` | Tables 1, 4, 5 |
| Zoom-in | `build_maps.py`, `zoom_in.py`, `summarize_zoom.py` | Tables 2, 3 |
| Faithfulness | `build_maps.py --task coco`, `faithfulness.py`, `summarize_faithfulness.py` | Fig. 4 |

[docs/REPRODUCE.md](docs/REPRODUCE.md) gives every command, the expected numbers and the compute needed. On our attention dumps, `evaluate_localization.py` reproduces the WhereBits row and nine baseline rows of Table 1 exactly (the post-hoc rows and the three label-tuned heads are not part of this release). `data/paper_questions/` lists the question ids of every benchmark and the calibration/evaluation split.

## Repository layout

```
wherebits/
  core.py            the estimator: token scores, where-to-look bits, head weights, layer maps, prior, final map
  crop.py            crop rules: primary (map sum) and bits window
  pipeline.py        WhereBitsVLM: image + question -> map and crop for a Hugging Face VLM
  models.py          model registry, frames, prompts
  capture.py         answer-start attention rows with vlmprobe
  dump.py            attention dump format
  offline.py         maps and baselines from dumps
  prior.py           shipped priors (priors/*.json): loading and writing
  metrics.py         pointing, token AUROC, VQA accuracy
  zoom.py            zoom-in answers (generation and forced choice)
  faithfulness.py    deletion / insertion; codec.py: bit axis from a learned codec
  probe_baselines.py Chefer relevance, Grad-CAM, rollout, token activation map
  stats.py           bootstrap, Wilcoxon, Holm, sign-flip tests
scripts/             data preparation, dumps, calibration, evaluation, summaries, demo
slurm/array.sbatch   template for sharded jobs on a SLURM cluster
data/paper_questions question ids and splits of the paper
tests/               unit tests (numpy only)
```

| Paper | Code |
|---|---|
| Eq. (1), per-head token score `d` | `core.token_scores` |
| Prop. 2, whether- and where-to-look bits | `core.whether_bits`, `core.where_bits` |
| Eq. (3), head weight `w = [B^x − B^∅]_+` | `core.head_weights` |
| Layer map `F_l`, layer bits `W_l` | `core.layer_maps`, `WhereBitsResult.layer_bits` |
| Eq. (4), prior `π` and map `M` | `core.layer_prior`, `core.wherebits` |
| Crop rules (Sec. 3.5) | `crop.best_window`, `crop.bits_window` |

## Citation

```bibtex
@misc{fu2026bitsnotboxes,
  title  = {Bits, Not Boxes: Label-Free Layer Selection for Visual Localization in VLMs},
  author = {Fu, Haisheng and Huang, Gexin and Zhao, Haikai and Liang, Jie and Fang, Zhenman and Han, Jingning and Li, Xiaoxiao},
  year   = {2026},
  note   = {Preprint}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). It depends on vlmprobe, which has its own license (PolyForm Noncommercial 1.0.0 and MIT), and the models and benchmarks keep their own licenses and terms of use.

## Acknowledgments

Attention is read with [vlmprobe](https://github.com/gexinh/vlm-probe) (see its license), codec bits come from [CompressAI](https://github.com/InterDigitalInc/CompressAI), and the models are loaded with [Hugging Face Transformers](https://github.com/huggingface/transformers). The benchmarks (COCO, RefCOCO, TextVQA, V\* Bench) are downloaded from their sources and keep their own licenses.
