---
license: cc-by-nc-sa-4.0
task_categories:
  - object-detection
  - video-text-to-text
language:
  - en
pretty_name: VISTA
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: test
        path: data/test-*.parquet
---

# VISTA: A Benchmark for Spatio-Temporal Interaction Understanding in Videos

VISTA is an evaluation benchmark for **spatio-temporal video grounding (STVG)** with vision-language models: given a trimmed video and a text query, localise the referred entity with one box per frame. Its contribution is an **interaction-centric taxonomy** over existing STVG / referring-video datasets, so that grounding performance can be broken down by who interacts, whether the query needs spatial or temporal reasoning, and what kind of interaction it is.

Paper: CVPR 2026 Workshop (PVUW) ([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/PVUW/html/Aparcedo_VISTA_A_Benchmark_for_Spatio-Temporal_Interaction_Understanding_in_Videos_CVPRW_2026_paper.html), [arXiv:2605.01391](https://arxiv.org/abs/2605.01391)). Code: [github.com/aaparcedo/VISTA](https://github.com/aaparcedo/VISTA).

## What is (and is not) in this release

- **Included:** annotations only. For each sample: source dataset and video identifier, frame range, ground-truth box per frame, the freeform query, the referral query (where one exists), and the VISTA taxonomy labels.
- **Not included:** videos or frames. VISTA re-uses videos from six existing datasets, several of which do not permit redistribution. Download them from their original sources (links below) and point the evaluation code at them.

## Composition

11,814 (sample, query) pairs drawn from the evaluation splits of six datasets:

| Source dataset | Split | Freeform queries | Referral queries | Total |
|---|---|---|---|---|
| HC-STVG v1 | test | 1,159 | 1,141 | 2,300 |
| HC-STVG v2 | val | 2,000 | 2,000 | 4,000 |
| VidSTG | test (quality-filtered) | 1,144 | 1,144 | 2,288 |
| VidVRD | test (quality-filtered) | 888 | 888 | 1,776 |
| MeViS | valid_u, single-object expressions | 616 | – | 616 |
| Ref-YouTube-VOS | valid | 834 | – | 834 |
| **Total** | | 6,641 | 5,173 | **11,814** |

- A *sample* is one target tube in one video segment. For VidSTG, the same target and caption can appear in several distinct segments (different frame ranges); these are distinct samples.
- 18 HC-STVG v1 clips have no referral query.
- MeViS and Ref-YouTube-VOS provide masks; VISTA boxes for them are derived from those masks.

## How it was built

1. **Sample selection.** Evaluation splits of the six datasets. VidSTG and VidVRD were reduced to quality-filtered subsets (1,144 segments, 888 relation instances).
2. **Freeform queries.** The original human-written captions for HC-STVG, VidSTG, MeViS and Ref-YouTube-VOS. For VidVRD, which provides ⟨subject, predicate, object⟩ triplets, sentences were generated with **gpt-4o** (prompt in the paper's supplementary, Table 4).
3. **Referral queries.** Short subject-identifying phrases extracted from the freeform query with **gpt-4o** (supplementary, Table 5 for VidVRD).
4. **Taxonomy labels** (freeform queries only), assigned by **gpt-4o-mini** with few-shot prompts (supplementary, Tables 1–3):
   - *Spatio-temporal* (1 label): Spatial vs. Temporal.
   - *Involved entities* (1 label): Human–Human, Human–Object, Human–Animal, Animal–Animal, Animal–Object, Object–Object, Human–Self, No Interaction.
   - *Fine-grained interaction* (multi-label): Affective, Social, Supportive, Physical, Relational Movement, Cooperative, Competitive, Antagonistic, Observation, Communicative, Proximity, Body Motion, Provisioning, Passive.
5. **Manual review.** Labels were reviewed and corrected by hand after each classification step. A small number of queries remained flagged for review and are released as labelled.

### Label quality
On a stratified sample of 113 queries labelled by two humans: human–human Cohen's κ = 0.98 (entity), 0.77 (spatio-temporal), 0.83 (fine-grained). Human–GPT agreement is lower: 0.76 / 0.69 / 0.67, as reported in the paper. GPT errors concentrate on visually ambiguous or under-specified captions, which is why the manual pass exists. Labels are assigned **per caption text**: identical captions on different clips share labels.

## Files and fields
One split, `test`: 6,641 records (one per sample), 11,814 (sample, query) pairs. On the Hub the file is `data/test-00000-of-00001.parquet`, where `gt_boxes` is a list of `{frame, box}`; the GitHub repo ships the same data as `data/vista_annotations.jsonl` with `gt_boxes` as `{frame: box}`.

| Field | Meaning |
|---|---|
| `sample_id` | `<dataset>-<source id>`; unique |
| `source_dataset`, `split` | e.g. `hcstvg1`, `test` |
| `video_id`, `video_relpath` | path relative to the source dataset's own root (videos for HC-STVG/VidSTG/VidVRD; JPEG frame folders for MeViS/RVOS) |
| `width`, `height` | from source metadata (null for MeViS/RVOS) |
| `start_frame`, `end_frame`, `gt_boxes` | ground-truth tube: frame index → `[x1, y1, x2, y2]` pixels. MeViS frame keys index the sorted frame list; RVOS frames are pre-sampled every 5th |
| `query_freeform`, `query_freeform_origin` | `original` caption, or `gpt-4o` (VidVRD) |
| `query_referral`, `query_referral_origin` | null for MeViS/RVOS and 18 HC-STVG v1 clips; origin `gpt-4o` or `existing-refined-sentences` (HC-STVG v1) |
| `label_st` | `Spatial` or `Temporal` |
| `label_entity` | one of the 8 entity classes |
| `labels_fine` | fine-grained interaction labels (multi-label, 14 classes) |

## Evaluation
Models receive every 5th frame of the ground-truth segment (Ref-YouTube-VOS is already sampled at that rate) and must output one box per frame. **mvIoU** is the mean per-frame IoU over the sampled frames. For VidSTG and VidVRD a sampled frame without a GT box scores 0. Code, protocol details and per-model results: [github.com/aaparcedo/VISTA](https://github.com/aaparcedo/VISTA).

## Licensing
VISTA annotations: **CC BY-NC-SA 4.0**.

**The original datasets' own terms also apply to their videos and annotations:**

| Source | Terms stated by the source |
|---|---|
| MeViS (videos largely from MOSE) | CC BY-NC-SA 4.0; non-commercial research only |
| Ref-YouTube-VOS (YouTube-VOS) | annotations CC BY 4.0; data for non-commercial research only |
| HC-STVG v1/v2 | no license stated |
| VidSTG (videos: VidOR, from YFCC100M) | no license stated; YFCC100M items carry individual CC licenses |
| VidVRD (videos: ILSVRC2016-VID) | no license stated; ImageNet terms may apply |

## Limitations
- Built on existing datasets; inherits their biases (e.g. HC-STVG is human-centric movie footage).
- Taxonomy labels are LLM-generated with manual review, not exhaustive human annotation; agreement figures above.
- Freeform queries for VidVRD and all referral queries are LLM-generated.
- Category frequencies are highly unbalanced; conclusions about rare categories are weak.
- 4 queries have no spatio-temporal label, 3 no entity label and 3 no fine-grained label (null in the release). One HC-STVG v2 sample (`hcstvg2-195_8oL0i5WorkE`) has an invalid bounding box in the source annotation and is released unchanged.

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
