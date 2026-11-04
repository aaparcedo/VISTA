"""Box utilities. Identical arithmetic to the code used for the paper."""


def box_iou(box1, box2):
    """IoU of two boxes in [x1, y1, x2, y2] pixels; 0 if the union is empty (e.g. both boxes empty)."""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / union if union > 0 else 0


def rescale_box_from_1000(box, width, height):
    """Map a model box on the 0-1000 normalised grid to pixel corners, truncated to ints and clamped to the frame."""
    x1, y1, x2, y2 = (v / 1000.0 for v in box)
    return [max(0, int(x1 * width)), max(0, int(y1 * height)), min(width, int(x2 * width)), min(height, int(y2 * height))]
