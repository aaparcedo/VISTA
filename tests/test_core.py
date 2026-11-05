"""Offline tests: box metrics, frame sampling rules, and Table-1 aggregation. No GPU or dataset needed.

    uv pip install pytest opencv-python-headless && pytest -q
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vista.data import VISTADataset  # noqa: E402
from vista.metrics import box_iou, rescale_box_from_1000  # noqa: E402


# ---------- metrics ----------
def test_iou_identical_and_disjoint():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0


def test_iou_half_overlap():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_iou_empty_boxes_score_zero():
    # a sampled frame without GT is scored against [0, 0, 0, 0]: always IoU 0
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    assert box_iou([1, 1, 5, 5], [0, 0, 0, 0]) == 0


def test_rescale_from_1000_grid():
    assert rescale_box_from_1000([0, 0, 1000, 1000], 640, 480) == [0, 0, 640, 480]
    assert rescale_box_from_1000([500, 250, 750, 1200], 640, 480) == [320, 120, 480, 480]  # clamped, truncated


# ---------- sampling ----------
def _write(tmp, samples):
    p = os.path.join(tmp, "ann.jsonl")
    with open(p, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return p


def _paths(tmp, **roots):
    p = os.path.join(tmp, "paths.yaml")
    with open(p, "w") as f:
        f.write("video_roots:\n" + "".join(f"  {k}: {v}\n" for k, v in roots.items()))
    return p


def _video(path, n=30, w=64, h=48):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), i, np.uint8))
    vw.release()


def _sample(ds, gt, start, end, rel, ref="the man"):
    return {"sample_id": f"{ds}-x", "source_dataset": ds, "video_relpath": rel, "start_frame": start, "end_frame": end,
            "gt_boxes": gt, "query_freeform": "the man walks", "query_referral": ref}


def test_vidstg_samples_every_5th_and_scores_missing_gt_as_empty(tmp_path):
    _video(str(tmp_path / "v.avi"))
    gt = {str(f): [f, f, f + 5, f + 5] for f in range(2, 20) if f != 12}       # frame 12 has no GT
    ds = VISTADataset(_write(tmp_path, [_sample("vidstg", gt, 2, 19, "v.avi")]), _paths(tmp_path, vidstg=str(tmp_path)), "vidstg", "freeform")
    _, frames, boxes = ds[0]
    assert len(frames) == len(boxes) == 4                                        # frames 2, 7, 12, 17
    assert boxes == [[2, 2, 7, 7], [7, 7, 12, 12], [0, 0, 0, 0], [17, 17, 22, 22]]


def test_hcstvg_keeps_only_frames_inside_trajectory(tmp_path):
    _video(str(tmp_path / "v.avi"))
    gt = {str(f): [f, 0, f + 1, 1] for f in range(3, 14)}                        # trajectory frames 3..13
    ds = VISTADataset(_write(tmp_path, [_sample("hcstvg1", gt, 3, 20, "v.avi")]), _paths(tmp_path, hcstvg1=str(tmp_path)), "hcstvg1", "freeform")
    _, frames, boxes = ds[0]
    assert [b[0] for b in boxes] == [3, 8, 13]                                   # 18 is outside the trajectory: dropped


def test_referral_task_skips_samples_without_referral(tmp_path):
    _video(str(tmp_path / "v.avi"))
    s1, s2 = _sample("hcstvg1", {"0": [0, 0, 1, 1]}, 0, 0, "v.avi"), _sample("hcstvg1", {"0": [0, 0, 1, 1]}, 0, 0, "v.avi", ref=None)
    s2["sample_id"] = "hcstvg1-y"
    ds = VISTADataset(_write(tmp_path, [s1, s2]), _paths(tmp_path, hcstvg1=str(tmp_path)), "hcstvg1", "referral")
    assert len(ds) == 1 and ds.samples[0]["query"] == "the man"


@pytest.mark.parametrize("dataset,gt_ids,kept_ids,image_index", [
    ("mevis", range(0, 11), [0, 5, 10], [0, 5, 10]),   # every 5th GT entry; image = sorted_files[frame_id]
    ("rvos", range(0, 11, 5), [0, 5, 10], [0, 1, 2]),   # pre-sampled ids, all kept; image = sorted_files[position]
])
def test_image_datasets(tmp_path, dataset, gt_ids, kept_ids, image_index):
    folder = tmp_path / "vid"
    folder.mkdir()
    for i in range(12):
        cv2.imwrite(str(folder / f"{i:05d}.png"), np.full((8, 8, 3), i * 20, np.uint8))  # pixel value encodes file index
    gt = {str(i): [i, 0, i + 1, 1] for i in gt_ids}
    ds = VISTADataset(_write(tmp_path, [_sample(dataset, gt, 0, 10, "vid", ref=None)]), _paths(tmp_path, **{dataset: str(tmp_path)}), dataset, "freeform")
    _, frames, boxes = ds[0]
    assert [b[0] for b in boxes] == kept_ids
    assert [int(f[0, 0, 0]) // 20 for f in frames] == image_index


# ---------- aggregation ----------
def test_make_table1_counts_each_pair_once(tmp_path):
    ann = [{"sample_id": "a", "source_dataset": "hcstvg1", "label_st": "Spatial", "label_entity": "Human-Human", "labels_fine": ["Social"],
            "query_freeform": "q", "query_referral": "r"},
           {"sample_id": "b", "source_dataset": "hcstvg1", "label_st": "Temporal", "label_entity": "Human-Object", "labels_fine": ["Physical"],
            "query_freeform": "q", "query_referral": None}]
    annp = _write(tmp_path, ann)
    res = tmp_path / "res"
    res.mkdir()
    rows = [("a", "freeform", 0.8), ("b", "freeform", 0.4), ("a", "referral", 0.6), ("a", "referral", 0.6)]   # last row is a duplicate
    with open(res / "x.jsonl", "w") as f:
        for sid, t, v in rows:
            f.write(json.dumps({"sample_id": sid, "model": "m", "dataset": "hcstvg1", "task_type": t, "mvIoU": v}) + "\n")
    out = tmp_path / "tables"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts/make_table1.py"), "--results", str(res),
                    "--annotations", annp, "--out", str(out)], check=True, capture_output=True)
    import pandas as pd
    t = pd.read_csv(out / "table1.csv", index_col=0).loc["m"]
    assert t["R"] == 60.0 and t["F"] == 60.0 and t["R&F"] == pytest.approx(60.0)
    assert t["S"] == 80.0 and t["T"] == 40.0 and t["HH"] == 80.0 and t["HO"] == 40.0


# ---------- batching ----------
class _FakeRunner:
    """Box depends only on the frame's pixel value, so batched and per-frame calls must agree."""
    def run_inference(self, image, query):
        v = int(np.asarray(image)[0, 0, 0])
        box = np.array([[v, 0, v + 100, 100]])
        return "", [type("T", (), {"cpu": lambda s: type("N", (), {"numpy": lambda s2: box})()})()], query, f"r{v}"

    def run_inference_batch(self, images, query):
        return [self.run_inference(im, query) for im in images]


def test_batched_evaluation_matches_per_frame():
    from vista.run_eval import evaluate_sample
    frames = [np.full((10, 20, 3), i * 10, np.uint8) for i in range(7)]
    frames[3] = None                                                              # unreadable frame stays a miss
    gt = [[0, 0, 2, 1]] * 7
    one = evaluate_sample(frames, gt, "q", _FakeRunner(), batch_size=1)
    many = evaluate_sample(frames, gt, "q", _FakeRunner(), batch_size=4)
    for k in ("predicted_boxes", "responses", "mvIoU"):
        assert one[k] == many[k]
    assert one["predicted_boxes"][3] == [0, 0, 0, 0] and one["responses"][3] is None
