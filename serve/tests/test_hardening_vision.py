"""Image limits the pixel budget and the source policy leave open: a side, the converted file, and the C++ constants."""
import io
import re
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from serve import ROOT
from serve.engine import vision
from serve.engine.vision import Vision
from serve.tests.test_vision_pixel_budget import tiff


def header_constant(name: str) -> int:
    """A constant of include/basalt/vision/preprocess.hpp, as the decoder sees it."""
    text = (ROOT / "include/basalt/vision/preprocess.hpp").read_text(encoding="utf-8")
    m = re.search(rf"\b{name}\s*=\s*([^;]+);", text)
    assert m, f"{name} is not in preprocess.hpp"
    value = m.group(1).strip()
    sizes = re.fullmatch(r"(\d+)(?:u|ll)?\s*<<\s*(\d+)", value)
    return (int(sizes.group(1)) << int(sizes.group(2))) if sizes else int(value)


class SideLimitTest(unittest.TestCase):
    def test_a_side_past_the_cap_is_refused_before_the_pixels_load(self):
        # both are inside the pixel budget: 32769 and 64 Mi pixels
        for w, h in ((vision.IMAGE_MAX_SIDE + 1, 1), (1, vision.IMAGE_MAX_SIDE + 1), (1 << 26, 1)):
            with self.subTest(size=(w, h)), self.assertRaisesRegex(ValueError, "a side may not pass"):
                Vision.normalize(tiff(w, h))

    def test_the_cap_itself_is_not_a_size_refusal(self):
        # the header carries no pixels, so the read fails later and for another reason
        with self.assertRaisesRegex(ValueError, "could not be read"):
            Vision.normalize(tiff(vision.IMAGE_MAX_SIDE, 1))


class ConvertedFileTest(unittest.TestCase):
    @staticmethod
    def tiff_image() -> bytes:
        from PIL import Image
        out = io.BytesIO()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(out, format="TIFF")
        return out.getvalue()

    def test_a_conversion_past_the_byte_cap_is_refused(self):
        with patch.object(vision, "IMAGE_MAX_BYTES", 16), self.assertRaisesRegex(ValueError, "once converted"):
            Vision.normalize(self.tiff_image())

    def test_a_conversion_inside_the_cap_is_a_png(self):
        self.assertEqual(Vision.normalize(self.tiff_image())[:8], b"\x89PNG\r\n\x1a\n")

    def test_the_cap_is_met_exactly(self):
        size = len(Vision.normalize(self.tiff_image()))
        with patch.object(vision, "IMAGE_MAX_BYTES", size):
            self.assertEqual(len(Vision.normalize(self.tiff_image())), size)
        with patch.object(vision, "IMAGE_MAX_BYTES", size - 1), self.assertRaises(ValueError):
            Vision.normalize(self.tiff_image())


    def test_conversion_stops_before_oversized_chunk_is_stored(self):
        from PIL import Image
        data = self.tiff_image()
        reached_after_write = []

        def save(image, output, **kwargs):
            self.assertEqual(kwargs["format"], "PNG")
            output.write(b"abcd")
            try:
                output.write(b"efghi")
            finally:
                self.assertEqual(output.getvalue(), b"abcd")
            reached_after_write.append(True)

        with patch.object(vision, "IMAGE_MAX_BYTES", 8), patch.object(Image.Image, "save", save):
            with self.assertRaisesRegex(ValueError, "once converted"):
                Vision.normalize(data)
        self.assertEqual(reached_after_write, [])


class GrowingFileTest(unittest.TestCase):
    def test_file_growing_after_stat_is_read_with_a_bound_and_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "image"
            path.write_bytes(b"abcd")
            opened, reads = [], []
            original_open = Path.open

            def grow_then_open(candidate, *args, **kwargs):
                # Simulate another writer after the size check, before the read.
                with original_open(candidate, "ab") as writer:
                    writer.write(b"e" * 100)
                stream = original_open(candidate, *args, **kwargs)
                opened.append(stream)
                wrapper = Mock(wraps=stream)
                wrapper.read.side_effect = lambda count=-1: (reads.append(count), stream.read(count))[1]
                class Context:
                    def __enter__(self):
                        return wrapper
                    def __exit__(self, *unused):
                        stream.close()
                return Context()

            with patch.object(Path, "open", grow_then_open):
                with self.assertRaisesRegex(ValueError, "larger than 4 bytes"):
                    vision.load_image(str(path), vision.ImageSources(dirs=(root,), max_bytes=4))
            self.assertEqual(reads, [5])
            self.assertTrue(opened and all(stream.closed for stream in opened))

    def test_allowed_file_at_limit_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "image"
            path.write_bytes(b"abcd")
            self.assertEqual(vision.load_image(str(path), vision.ImageSources(dirs=(root,), max_bytes=4)), b"abcd")


class DecoderConstantsTest(unittest.TestCase):
    """One budget, two decoders: the Pillow path here and stb in the encoder must refuse the same images."""

    def test_pixels(self):
        self.assertEqual(vision.IMAGE_MAX_PIXELS, header_constant("kMaxImagePixels"))

    def test_side(self):
        self.assertEqual(vision.IMAGE_MAX_SIDE, header_constant("kMaxImageSide"))

    def test_bytes(self):
        self.assertEqual(vision.IMAGE_MAX_BYTES, header_constant("kMaxImageBytes"))


if __name__ == "__main__":
    unittest.main()
