"""Conservative contours with explicit topology and raster fidelity checks."""
import cv2
import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon
from .common import require, binary_counts, pixel_metrics


def polygon_parts(geometry):
    if geometry.geom_type == 'Polygon':
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def parse_labels(text):
    polygons = []
    for line in text.splitlines():
        values = line.split()
        require(values and values[0] == '0', 'Labels must contain only class 0')
        require(len(values) >= 7 and len(values) % 2 == 1, 'Need at least three coordinate pairs')
        points = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
        require(np.isfinite(points).all() and (points >= 0).all() and (points <= 1).all(), 'Invalid normalized coordinates')
        shape = Polygon(points)
        require(shape.is_valid and shape.area > 0, 'Invalid or zero-area polygon')
        polygons.append(points)
    require(bool(polygons), 'Empty labels are not allowed for this positive-only dataset')
    return polygons


def rasterize(polygons, shape):
    # Match Ultralytics polygon2mask: float32 scaling, then int32 truncation.
    h, w = shape
    result = np.zeros((h, w), dtype=np.uint8)
    for polygon in polygons:
        points = (polygon * np.array([w, h], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(result, [points], color=1)
    return result.astype(bool)


def mask_to_labels(mask, minimum_iou=.98):
    h, w = mask.shape
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    require(hierarchy is not None, 'Empty source mask')
    require(not any(item[3] >= 0 for item in hierarchy[0]), 'Mask has holes: YOLO rows cannot encode holes; conversion refuses to fill them')
    raw, repaired = [], 0
    for contour in contours:
        points = contour.reshape(-1, 2)
        require(len(points) >= 3, 'Isolated one-pixel/linear component cannot form a valid polygon')
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
            repaired += 1
        pieces = list(polygon_parts(polygon))
        require(bool(pieces), 'Component has no positive-area polygon')
        for piece in pieces:
            require(piece.is_valid and piece.area > 0 and not piece.interiors, 'Unsupported polygon topology')
            raw.append(np.asarray(piece.exterior.coords[:-1], dtype=np.float64))
    # Pixel-center coordinates (+0.5) preserve integer rasterization and avoid
    # a systematic half-pixel bias when YOLO interpolates segment vertices.
    # No curve smoothing or contour approximation epsilon is applied.
    lines = ['0 ' + ' '.join(f'{v:.9f}' for v in ((p + .5) / [w, h]).flatten()) for p in raw]
    text = '\n'.join(lines) + '\n'
    polygons = parse_labels(text)
    reconstructed = rasterize(polygons, mask.shape)
    counts = binary_counts(reconstructed, mask)
    metrics = pixel_metrics(counts)
    require(counts['fp'] == 0, 'Conversion would introduce foreground across a gap/occluder')
    require(metrics['iou'] >= minimum_iou, f'Polygon fidelity {metrics["iou"]:.6f} below {minimum_iou}; inspect thin tips')
    return text, reconstructed, {**counts, **metrics, 'polygons': len(polygons),
                                 'vertices': sum(len(p) for p in polygons), 'repaired_contours': repaired}
