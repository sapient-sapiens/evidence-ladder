import unittest

from PIL import Image, ImageDraw

from src.visual_damage import registry_crop_is_obscured


class VisualDamageTest(unittest.TestCase):
    def test_saturated_dark_registry_overlay_is_obscured(self):
        image = Image.new("RGB", (612, 792), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((430, 320, 575, 530), fill=(210, 190, 60))
        draw.rectangle((465, 370, 545, 500), fill=(90, 110, 160))
        self.assertTrue(registry_crop_is_obscured(image))

    def test_blank_page_is_not_obscured(self):
        self.assertFalse(
            registry_crop_is_obscured(Image.new("RGB", (612, 792), "white"))
        )
