import unittest

import cv2
import numpy as np

from src.visual_risk_marks import (
    _BIOHAZARD_REFERENCE,
    _is_biohazard_contour,
    purple_triangle_in_image,
    red_biohazard_in_image,
)


class VisualRiskMarkTests(unittest.TestCase):
    def test_detects_large_purple_triangle_in_form_body(self):
        image = np.full((792, 612, 3), 255, dtype=np.uint8)
        cv2.polylines(
            image,
            [np.asarray([[260, 430], [210, 530], [310, 530]], dtype=np.int32)],
            True,
            (170, 90, 145),
            12,
        )
        self.assertTrue(purple_triangle_in_image(image))

    def test_rejects_purple_circle(self):
        image = np.full((792, 612, 3), 255, dtype=np.uint8)
        cv2.circle(image, (260, 500), 55, (170, 90, 145), 12)
        self.assertFalse(purple_triangle_in_image(image))

    def test_reference_red_biohazard_contour_matches(self):
        self.assertTrue(_is_biohazard_contour(_BIOHAZARD_REFERENCE))

    def test_plain_red_circle_is_not_biohazard_seal(self):
        contour = cv2.ellipse2Poly((400, 400), (45, 45), 0, 0, 360, 5)
        self.assertFalse(_is_biohazard_contour(contour.reshape(-1, 1, 2)))


if __name__ == "__main__":
    unittest.main()
