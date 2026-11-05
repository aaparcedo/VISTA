# VISTA: A Benchmark for Spatio-Temporal Interaction Understanding in Videos

Evaluation code and annotations for the VISTA benchmark (CVPR 2026 Workshop, PVUW).
Paper: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/PVUW/html/Aparcedo_VISTA_A_Benchmark_for_Spatio-Temporal_Interaction_Understanding_in_Videos_CVPRW_2026_paper.html) · [arXiv:2605.01391](https://arxiv.org/abs/2605.01391) · Dataset: [Hugging Face `aaparcedo01/VISTA`](https://huggingface.co/datasets/aaparcedo01/VISTA) · Project page: https://aaparcedo.github.io/VISTA/

VISTA evaluates vision-language models on **spatio-temporal video grounding**: given a video segment and a text query, predict one box per frame for the referred entity. Every query carries an interaction-centric taxonomy label (spatial vs. temporal, involved entities, fine-grained interaction type), so results can be broken down by interaction.

## What's here

| Path | Contents |
|---|---|
| `data/vista_annotations.jsonl` | 6,641 samples / 11,814 (sample, query) pairs: GT tubes, freeform + referral queries, taxonomy labels. Field reference in `DATASET_CARD.md` |
| `vista/` | data loading (`data.py`), model wrappers (`models.py`), evaluation (`run_eval.py`), box metrics (`metrics.py`) |
| `scripts/make_table1.py` | builds Table 1 and the fine-grained table from per-sample results |
| `configs/paths.example.yaml` | where your copies of the source datasets live |
| `env/` | per-model environment pins (see `env/README.md`) |

**Videos are not included.** VISTA re-uses six existing datasets; download them from their sources:
[HC-STVG v1/v2](https://github.com/tzhhhh123/HC-STVG) ·
[VidSTG](https://github.com/Guaranteer/VidSTG-Dataset) (videos: [VidOR](https://xdshang.github.io/docs/vidor.html)) ·
[VidVRD](https://xdshang.github.io/docs/imagenet-vidvrd.html) ·
[MeViS](https://github.com/henghuiding/MeViS) (valid_u) ·
[Ref-YouTube-VOS](https://youtube-vos.org/dataset/rvos/) (valid).

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/aaparcedo/VISTA.git && cd VISTA
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install -r env/<model-family>.txt        # see env/README.md
cp configs/paths.example.yaml configs/paths.yaml   # then edit the dataset roots
```

Model weights download from Hugging Face on first use. To use local copies, set `VISTA_<MODEL>_PATH`
(e.g. `VISTA_QWEN3VL_PATH`, `VISTA_MIMOVL_PATH`, `VISTA_INTERNVL25_PATH`, `VISTA_QWENVL_PATH`, `VISTA_SHIKRA_PATH`, `VISTA_FERRET_PATH`).
Shikra and Ferret also need their repositories cloned into the working directory (`shikra/`, `ml_ferret/`); MiniGPT-v2 needs a clone of
[Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) (`VISTA_MINIGPT4_REPO`), Llama-2-7b-chat (`VISTA_LLAMA2_CHAT_PATH`) and the
stage-3 checkpoint (`VISTA_MINIGPTV2_CKPT`).

## Evaluate

One model, one dataset, one query type:

```bash
python -m vista.run_eval --model qwen3vl --dataset hcstvg1 --task_type freeform \
    --annotations data/vista_annotations.jsonl --paths configs/paths.yaml \
    --output results/qwen3vl/hcstvg1_freeform.jsonl
```

- Models: `cogvlm`, `shikra`, `ferret`, `qwen3vl`, `mimovl`, `internvl2.5`, `qwenvl`, `gdino`, `minigptv2`.
- Datasets: `hcstvg1`, `hcstvg2`, `vidstg`, `vidvrd`, `mevis`, `rvos`. Query types: `freeform`, `referral` (MeViS and Ref-YouTube-VOS are freeform only).
- Output is resumable JSON Lines keyed by `sample_id`. Split long runs across GPUs with `--shard_index i --num_shards N`.

Full benchmark for one model = 10 runs (6 freeform + 4 referral).

## Build the tables

```bash
python scripts/make_table1.py --results results/ --annotations data/vista_annotations.jsonl --out tables/
```

Writes `tables/table1.csv`, `tables/table_finegrained.csv` and `tables/coverage.json`, and warns if any model has fewer than 11,814 pairs.

## Tests

`uv pip install pytest opencv-python-headless && pytest -q` runs offline checks of the box metric, the per-dataset frame-sampling rules (on synthetic videos and frame folders), batched vs. per-frame evaluation and the table builder. No GPU or dataset needed.

## Protocol details (read before comparing numbers)

- **Frames.** Videos are trimmed to the ground-truth segment and every 5th frame is sent to the model (Ref-YouTube-VOS frames are already sampled at that rate). The model predicts one box per frame.
- **Metric.** `mvIoU` is the mean per-frame IoU over the sampled frames, averaged over samples and reported ×100.
- **Missing GT frames (VidSTG, VidVRD).** A sampled frame without a GT box is scored against an empty box, i.e. IoU 0.
- **Undecodable frames.** A sampled frame the video decoder cannot read is skipped.
- **Model outputs.** All wrappers return boxes on a 0–1000 normalised grid, rescaled to pixels. An unparsable answer counts as an empty box.
- **MiniGPT-v2 precision.** The wrapper follows the official evaluation config (Llama-2 loaded in 8-bit). Run it with `--batch_size 1`: in 8-bit, batching changes about a third of the answers.

## Models in the paper

| Model | Wrapper in this repo |
|---|---|
| CogVLM-grounding, Shikra, Ferret, Qwen3-VL, MiMo-VL | yes |
| InternVL2.5-8B, Qwen-VL-Chat, Grounding DINO (Swin-T) | yes (implemented from the official model cards / transformers) |
| MiniGPT-v2 | yes (`minigptv2`; official repo code path, env `env/minigptv2.txt`) |
| LLaVA-Grounding, Sphinx-v2 | not included: use the models' own repositories ([LLaVA-Grounding](https://github.com/UX-Decoder/LLaVA-Grounding), [SPHINX](https://github.com/Alpha-VLLM/LLaMA2-Accessory)) |

## Results

Table 1 of the paper (CVPR 2026 Workshop camera-ready; the same table appears in arXiv v2), mvIoU ×100.
R / F: referral / freeform queries; R&F: all pairs. S / T: spatial / temporal queries.
Entity columns: Animal–Animal (AA), Animal–Object (AO), Human–Animal (HA), Human–Human (HH), Human–Object (HO), Human–Self (HS), No Interaction (NI), Object–Object (OO).

| Model | R | F | R&F | S | T | AA | AO | HA | HH | HO | HS | NI | OO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ***Foundation model without LLM*** | | | | | | | | | | | | | |
| Grounding DINO (Swin-T) | 37.79 | 32.34 | 34.64 | 35.0 | 30.8 | 12.6 | 38.8 | 52.1 | 29.9 | 37.0 | 39.4 | 41.3 | 38.3 |
| ***Generalist MLLMs*** | | | | | | | | | | | | | |
| InternVL2.5-8B | 51.11 | 48.65 | 49.73 | 46.3 | 48.0 | 37.9 | 49.8 | 52.2 | 49.2 | 49.5 | 50.9 | 48.2 | 47.7 |
| MiniGPT-v2 | 46.62 | 45.13 | 45.78 | 43.1 | 44.3 | 33.6 | 47.4 | 48.5 | 46.0 | 46.1 | 46.4 | 44.3 | 43.1 |
| Sphinx-v2 | 47.79 | 44.28 | 45.82 | 42.6 | 45.0 | 30.4 | 46.9 | 51.0 | 47.0 | 46.8 | 48.1 | 46.0 | 42.5 |
| Qwen-VL-Chat | 45.56 | 45.43 | 45.49 | 45.7 | 45.3 | 33.7 | 54.8 | 65.2 | 48.2 | 47.9 | 58.8 | 42.2 | 31.7 |
| Qwen3-VL | 62.85 | 64.41 | 63.96 | 64.8 | 64.3 | 59.5 | 63.2 | 74.5 | 66.2 | 64.7 | 75.7 | 60.6 | 59.1 |
| MiMo-VL | 43.34 | 42.13 | 44.54 | 36.9 | 43.5 | 40.4 | 38.3 | 38.3 | 45.1 | 36.1 | 46.0 | 43.0 | 27.0 |
| ***Specialist MLLMs*** | | | | | | | | | | | | | |
| Shikra | 30.91 | 31.44 | 31.21 | 29.9 | 32.4 | 20.0 | 28.9 | 36.0 | 34.0 | 35.3 | 38.8 | 31.4 | 24.6 |
| Ferret-v1 | 17.74 | 22.71 | 20.53 | 20.9 | 23.8 | 14.9 | 23.7 | 23.3 | 26.4 | 24.5 | 33.6 | 19.7 | 13.4 |
| CogVLM-grounding | 60.56 | 50.13 | 54.70 | 57.5 | 45.7 | 48.1 | 60.3 | 70.2 | 44.7 | 54.0 | 46.8 | 50.7 | 54.0 |
| LLaVA-Grounding | 22.51 | 30.47 | 27.11 | 28.1 | 31.9 | 10.7 | 37.1 | 56.1 | 28.4 | 37.0 | 38.9 | 36.0 | 28.4 |

The fine-grained interaction results and the analysis are in the paper. To evaluate a model with this code, see [Evaluate](#evaluate) and [Build the tables](#build-the-tables).

## License

Code: MIT (`LICENSE`). Annotations: CC BY-NC-SA 4.0 (see `DATASET_CARD.md`). The source datasets' own terms apply to their videos.

## Citation

```bibtex
@InProceedings{Aparcedo_2026_CVPR,
    author    = {Aparcedo, Alejandro and Kumar, Akash and Garg, Aaryan and Pham, Dalton and Chen, Wen-Kai and Bharadwaj, Anirudh and Chadha, Aman and Rawat, Yogesh},
    title     = {VISTA: A Benchmark for Spatio-Temporal Interaction Understanding in Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {8411-8422}
}
```
The arXiv version ([2605.01391](https://arxiv.org/abs/2605.01391)) is titled "VISTA: Video Interaction Spatio-Temporal Analysis Benchmark".
