import sys
from pathlib import Path
import unittest
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.polygons import mask_to_labels,parse_labels,rasterize
from utils.common import binary_counts,pixel_metrics,connectivity,union_prediction


class PolygonTests(unittest.TestCase):
    def test_curved_disconnected_fragments_do_not_bridge_occluder(self):
        mask=np.zeros((120,200),np.uint8)
        cv2.polylines(mask,[np.array([[10,90],[40,85],[70,55],[85,30]])],False,1,3)
        cv2.polylines(mask,[np.array([[120,25],[140,45],[180,90]])],False,1,3)
        text,reconstructed,report=mask_to_labels(mask>0)
        self.assertEqual(len(parse_labels(text)),2)
        self.assertEqual(report['fp'],0)
        self.assertGreater(report['iou'],.98)
        self.assertFalse(reconstructed[:,95:110].any())
        self.assertEqual(connectivity(reconstructed)['connected_components'],2)

    def test_invalid_holes_empty_and_degenerate_masks_fail_explicitly(self):
        mask=np.zeros((20,20),np.uint8)
        with self.assertRaises(ValueError):mask_to_labels(mask>0)
        mask[3:17,3:17]=1;mask[8:12,8:12]=0
        with self.assertRaisesRegex(ValueError,'holes'):mask_to_labels(mask>0)
        mask[:]=0;mask[2,2]=1
        with self.assertRaisesRegex(ValueError,'one-pixel'):mask_to_labels(mask>0)
        with self.assertRaises(ValueError):parse_labels('1 0 0 1 0 1 1')
        with self.assertRaises(ValueError):parse_labels('0 0 0 1 1 0 1 1 0')
        with self.assertRaises(ValueError):parse_labels('0 0 0 1.2 0 1 1')

    def test_prediction_unions_fragments_without_resizing(self):
        from types import SimpleNamespace
        import torch
        masks=torch.zeros((2,8,12))
        masks[0,2:4,2:4]=1;masks[1,4:6,8:10]=1
        result=SimpleNamespace(orig_shape=(8,12),masks=SimpleNamespace(data=masks),boxes=SimpleNamespace(cls=torch.tensor([0.,0.])))
        union=union_prediction(result)
        self.assertEqual(union.sum(),8)
        self.assertEqual(connectivity(union)['connected_components'],2)
        result.masks=None
        self.assertFalse(union_prediction(result).any())
        result.masks=SimpleNamespace(data=torch.zeros((1,4,6)))
        with self.assertRaisesRegex(ValueError,'retina_masks'):union_prediction(result)

    def test_metrics_empty_and_union_semantics(self):
        truth=np.array([[True,False],[False,True]])
        self.assertEqual(pixel_metrics(binary_counts(truth,truth))['dice'],1)
        self.assertEqual(pixel_metrics(binary_counts(np.zeros_like(truth),truth))['iou'],0)
        self.assertEqual(pixel_metrics(binary_counts(np.zeros_like(truth),np.zeros_like(truth)))['iou'],1)
        c=binary_counts(np.array([[True,True],[False,False]]),truth)
        self.assertEqual(c,{'tp':1,'fp':1,'fn':1,'tn':1})
        self.assertAlmostEqual(pixel_metrics(c)['iou'],1/3)
        self.assertEqual(pixel_metrics(c)['dice'],.5)


if __name__=='__main__':unittest.main()
