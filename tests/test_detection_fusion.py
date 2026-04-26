import sys
import types
import unittest


class _DummyRFDetr:
    pass


module = types.ModuleType("rfdetr")
module.RFDETRMedium = _DummyRFDetr
module.RFDETRLarge = _DummyRFDetr
module.RFDETRSegPreview = _DummyRFDetr
module.RFDETRSegXLarge = _DummyRFDetr
module.RFDETRSeg2XLarge = _DummyRFDetr
module.RFDETR2XLarge = _DummyRFDetr
sys.modules.setdefault("rfdetr", module)

from utils import detection_pipeline


def _det(bbox, confidence=0.5, class_id=4):
    return {
        "class_id": class_id,
        "class_name": f"class_{class_id}",
        "confidence": confidence,
        "bbox": [float(v) for v in bbox],
        "source": "wide_slice",
    }


class DetectionFusionTest(unittest.TestCase):
    def test_merges_same_class_connected_overlap_chain(self):
        detections = [
            _det([0, 0, 100, 100], confidence=0.9),
            _det([40, 0, 140, 100], confidence=0.8),
            _det([120, 0, 220, 100], confidence=0.7),
        ]

        merged = detection_pipeline._merge_classwise_overlapping_boxes(detections, 0.5)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["bbox"], [0.0, 0.0, 220.0, 100.0])
        self.assertEqual(merged[0]["confidence"], 0.9)

    def test_keeps_different_classes_separate_when_only_partially_overlapping(self):
        detections = [
            _det([0, 0, 100, 100], confidence=0.9, class_id=4),
            _det([40, 0, 140, 100], confidence=0.8, class_id=7),
        ]

        merged = detection_pipeline._merge_classwise_overlapping_boxes(detections, 0.5)

        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
