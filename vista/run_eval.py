"""VISTA evaluation: run one model on one dataset / query type and write per-sample results.

Example:
    python -m vista.run_eval --model cogvlm --dataset hcstvg1 --task_type freeform \
        --annotations data/vista_annotations.jsonl --paths configs/paths.yaml \
        --output results/cogvlm_hcstvg1_freeform.jsonl

Output is JSON Lines, one record per sample (`sample_id`, `mvIoU`, per-frame boxes, raw model responses); the run resumes
from an existing output file. Shard large runs with --shard_index/--num_shards.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .data import VISTADataset
from .metrics import box_iou, rescale_box_from_1000


def evaluate_sample(frames, gt_boxes, query, runner, legacy_channel_swap=False, batch_size=1):
    """Query the model on every sampled frame; mvIoU = mean per-frame IoU over the sampled GT frames.

    With batch_size > 1 and a runner that has `run_inference_batch`, frames are sent in batches (same query)."""
    preds, responses, secs = [[0, 0, 0, 0]] * len(frames), [None] * len(frames), []
    valid = [i for i, f in enumerate(frames) if f is not None]  # an unreadable frame counts as a miss
    step = batch_size if batch_size > 1 and hasattr(runner, "run_inference_batch") else 1
    for b in range(0, len(valid), step):
        ids = valid[b:b + step]
        imgs = [Image.fromarray(frames[i] if legacy_channel_swap else cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in ids]
        t0 = time.time()
        outs = runner.run_inference_batch(imgs, query) if step > 1 else [runner.run_inference(imgs[0], query)]
        secs.append(time.time() - t0)
        for i, (_, boxes, _, response) in zip(ids, outs):
            h, w = frames[i].shape[:2]
            preds[i] = rescale_box_from_1000(boxes[0].cpu().numpy().flatten(), w, h) if boxes else [0, 0, 0, 0]
            responses[i] = response
    ious = [box_iou(p, g) for p, g in zip(preds, gt_boxes)]
    return {
        "mvIoU": float(np.mean(ious)) if ious else 0.0,
        "mvIoU03": float(np.mean([i >= 0.3 for i in ious])) if ious else 0.0,
        "mvIoU05": float(np.mean([i >= 0.5 for i in ious])) if ious else 0.0,
        "predicted_boxes": preds, "ground_truth_boxes": gt_boxes, "responses": responses,
        "inference_seconds": float(sum(secs)),
    }


def main(args):
    ds = VISTADataset(args.annotations, args.paths, args.dataset, args.task_type, args.frame_step)
    idx = [i for i in range(len(ds)) if i % args.num_shards == args.shard_index]
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            done = {json.loads(l)["sample_id"] for l in f if l.strip()}
        print(f"Resuming: {len(done)} samples already in {args.output}")
    legacy = args.legacy_channel_swap and args.dataset in ("mevis", "rvos")
    from .models import build_model  # imported here so the scoring code above works without torch
    runner = build_model(args.model)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    scores = []
    with open(args.output, "a") as out:
        for i in tqdm(idx, desc=f"{args.model}/{args.dataset}/{args.task_type}"):
            sample, frames, gt = ds[i]
            if sample["sample_id"] in done:
                continue
            assert frames, f"no frames for {sample['sample_id']}"
            r = evaluate_sample(frames, gt, sample["query"], runner, legacy, args.batch_size)
            rec = {"sample_id": sample["sample_id"], "model": args.model, "dataset": args.dataset,
                   "task_type": args.task_type, "query": sample["query"], "frame_step": args.frame_step,
                   "legacy_channel_swap": legacy, **r}
            out.write(json.dumps(rec) + "\n"); out.flush()
            scores.append(r["mvIoU"])
    if scores:
        print(f"mean mvIoU over {len(scores)} new samples: {100 * np.mean(scores):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VISTA evaluation")
    p.add_argument("--model", required=True, help="see vista.models.MODELS")
    p.add_argument("--dataset", required=True, choices=["hcstvg1", "hcstvg2", "vidstg", "vidvrd", "mevis", "rvos"])
    p.add_argument("--task_type", required=True, choices=["freeform", "referral"])
    p.add_argument("--annotations", default="data/vista_annotations.jsonl")
    p.add_argument("--paths", default="configs/paths.yaml", help="YAML with video_roots per dataset")
    p.add_argument("--output", required=True, help="JSON Lines output (resumable)")
    p.add_argument("--frame_step", type=int, default=5)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1, help="frames per model call, for wrappers with run_inference_batch")
    p.add_argument("--legacy-channel-swap", dest="legacy_channel_swap", action="store_true",
                   help="swap the red and blue channels of MeViS / Ref-YouTube-VOS frames before inference")
    main(p.parse_args())
