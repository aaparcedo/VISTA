"""VISTA data loading.

All samples come from one annotation file (data/vista_annotations.jsonl). Videos/frames are read from the
original datasets, whose root directories are given in a paths config (see configs/paths.example.yaml).

Frame sampling reproduces the loaders used for the paper exactly:
  * HC-STVG v1/v2: decode from `start_frame`, keep every `frame_step`-th frame of the GT tube.
  * VidSTG / VidVRD: decode from `start_frame` to `end_frame`, keep every `frame_step`-th frame. A kept frame with no GT box
    is scored against an empty box [0, 0, 0, 0] (IoU 0); this matches the original protocol.
  * MeViS: frames are JPEGs; GT frame ids index the sorted frame list; every `frame_step`-th GT frame is kept.
  * Ref-YouTube-VOS: frames are already sampled every 5th frame; all GT frames are kept (by position in the sorted list).
"""
import json
import os

import cv2
import yaml

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def load_paths(config_path):
    """Map dataset name -> root directory of its videos / frame folders."""
    with open(config_path) as f:
        return yaml.safe_load(f)["video_roots"]


def load_samples(annotation_path, dataset, task_type):
    """Samples of one dataset with a query for `task_type` ('freeform' or 'referral')."""
    samples = []
    with open(annotation_path) as f:
        for line in f:
            s = json.loads(line)
            if s["source_dataset"] != dataset:
                continue
            query = s["query_freeform"] if task_type == "freeform" else s["query_referral"]
            if query:
                samples.append(dict(s, query=query))
    return samples


class VISTADataset:
    def __init__(self, annotation_path, paths_config, dataset, task_type, frame_step=5):
        self.dataset = dataset
        self.root = load_paths(paths_config)[dataset]
        self.frame_step = frame_step
        self.samples = load_samples(annotation_path, dataset, task_type)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (sample, frames, gt_boxes): RGB-free BGR numpy frames and their [x1, y1, x2, y2] GT boxes, aligned."""
        s = self.samples[idx]
        if self.dataset in ("hcstvg1", "hcstvg2"):
            frames, boxes = self._video_frames(s, missing_box=None)
        elif self.dataset in ("vidstg", "vidvrd"):
            frames, boxes = self._video_frames(s, missing_box=[0, 0, 0, 0])
        elif self.dataset == "mevis":
            frames, boxes = self._image_frames(s, by_position=False, step=self.frame_step)
        elif self.dataset == "rvos":
            frames, boxes = self._image_frames(s, by_position=True, step=1)
        else:
            raise ValueError(f"unknown dataset {self.dataset}")
        return s, frames, boxes

    def _video_frames(self, s, missing_box):
        cap = cv2.VideoCapture(os.path.join(self.root, s["video_relpath"]))
        start, end = s["start_frame"], s["end_frame"]
        gt = s["gt_boxes"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        frames, boxes, cur = [], [], start
        while cap.isOpened() and cur <= end:
            ok, frame = cap.read()
            if not ok:
                break
            if (cur - start) % self.frame_step == 0:
                box = gt.get(str(cur), missing_box)
                if box is not None:  # HC-STVG keeps only frames inside the GT trajectory
                    frames.append(frame)
                    boxes.append(box)
            cur += 1
        cap.release()
        return frames, boxes

    def _image_frames(self, s, by_position, step):
        folder = os.path.join(self.root, s["video_relpath"])
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
        frames, boxes = [], []
        for i, (frame_id, box) in enumerate(s["gt_boxes"].items()):
            if i % step:
                continue
            frame = cv2.imread(os.path.join(folder, files[i if by_position else int(frame_id)]))
            if frame is None:
                raise FileNotFoundError(f"missing frame {frame_id} in {folder}")
            frames.append(frame)
            boxes.append(box)
        return frames, boxes
