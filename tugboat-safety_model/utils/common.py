from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / 'ai'
OUTPUTS = AI / 'outputs'
SPLITS = ('train', 'val', 'test')
DEFAULT_MODEL = 'yolo26n-seg.pt'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    require(bool(rows), 'No rows to save')
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def versions():
    return {'python': sys.version, 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in
                         ('ultralytics', 'torch', 'torchvision', 'opencv-python', 'numpy', 'Pillow', 'PyYAML', 'shapely')}}


def configure_runtime():
    # All application caches and model downloads remain in the repository.
    for variable, folder in [('YOLO_CONFIG_DIR', 'ultralytics'), ('MPLCONFIGDIR', 'matplotlib'), ('TORCH_HOME', 'torch')]:
        target = OUTPUTS / '.cache' / folder
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    os.environ['MPLBACKEND'] = 'Agg'
    from ultralytics import settings
    settings.update({'datasets_dir': str(ROOT), 'weights_dir': str(OUTPUTS / 'checkpoints'),
                     'runs_dir': str(OUTPUTS), 'sync': False})


def select_device(requested):
    import torch
    if requested != 'auto':
        if requested == 'mps':
            require(torch.backends.mps.is_available(), 'MPS is unavailable; use --device cpu')
        return requested
    if torch.backends.mps.is_available():
        return 'mps'
    return '0' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint, pretrained=False):
    from ultralytics import YOLO
    path = Path(checkpoint).expanduser()
    if pretrained and len(path.parts) == 1 and not path.exists():
        path = OUTPUTS / 'checkpoints' / path.name
        path.parent.mkdir(parents=True, exist_ok=True)
    if not pretrained:
        require(path.is_file(), f'Trained weights do not exist: {path}')
    require(path.suffix == '.pt', 'Use a .pt segmentation checkpoint, not an untrained YAML architecture')
    model = YOLO(str(path), task='segment')
    require(model.task == 'segment', 'Checkpoint is not a segmentation model')
    if not pretrained:
        require(model.names == {0: 'towline'}, f'Expected trained single-class towline model, got {model.names}')
    return model, path.resolve()


def new_output(name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Output name must be relative within ai/outputs')
    target = OUTPUTS / path
    target.mkdir(parents=True, exist_ok=False)
    return target


def read_mask(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'))
    require(np.array_equal(array[:, :, 0], array[:, :, 1]) and np.array_equal(array[:, :, 0], array[:, :, 2]), f'Non-grayscale mask: {path}')
    require(set(np.unique(array).tolist()) <= {0, 255}, f'Non-binary mask: {path}')
    return array[:, :, 0] == 255


def binary_counts(prediction, truth):
    require(prediction.shape == truth.shape, 'Mask dimensions differ')
    return {'tp': int(np.count_nonzero(prediction & truth)),
            'fp': int(np.count_nonzero(prediction & ~truth)),
            'fn': int(np.count_nonzero(~prediction & truth)),
            'tn': int(np.count_nonzero(~prediction & ~truth))}


def pixel_metrics(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    # Both empty = perfect IoU/Dice. Empty prediction against nonempty GT = 0.
    return {'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.,
            'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
            'pixel_precision': tp / (tp + fp) if tp + fp else (1. if not fn else 0.),
            'pixel_recall': tp / (tp + fn) if tp + fn else 1.}


def connectivity(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return {'connected_components': int(count - 1),
            'largest_component_fraction': float(areas.max() / areas.sum()) if len(areas) else 0.}


def union_prediction(result):
    h, w = result.orig_shape
    if result.masks is None:
        return np.zeros((h, w), dtype=bool)
    masks = result.masks.data.cpu().numpy()
    require(masks.shape[1:] == (h, w), 'Expected retina_masks=True original-size masks; refusing misaligned resize')
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return np.any(masks[classes == 0] > .5, axis=0) if np.any(classes == 0) else np.zeros((h, w), dtype=bool)


def green_overlay(image, mask):
    rgb = np.asarray(image.convert('RGB')).copy()
    rgb[mask] = (.45 * rgb[mask] + .55 * np.array([0, 255, 70])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (0, 255, 70), 1)
    return Image.fromarray(rgb)


def save_prediction(result, directory, stem):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mask = union_prediction(result)
    Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f'{stem}_mask.png')
    rgb = Image.fromarray(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
    green_overlay(rgb, mask).save(directory / f'{stem}_overlay.png')
    return mask


def contact_sheet(examples, output):
    # [(caption, RGB image, original mask bool, converted/predicted mask bool)]
    sheet = Image.new('RGB', (1152, 240 * len(examples)), '#14202a')
    for row, (caption, rgb, truth, predicted) in enumerate(examples):
        panels = [rgb.copy(), Image.fromarray(truth.astype(np.uint8) * 255).convert('RGB'), green_overlay(rgb, predicted)]
        for col, panel in enumerate(panels):
            panel.thumbnail((384, 216))
            sheet.paste(panel, (col * 384, row * 240 + 24))
        ImageDraw.Draw(sheet).text((8, row * 240 + 6), caption, fill='white')
    sheet.save(output)
