# -*- coding: utf-8 -*-
"""Unit-Tests für nn_imgLoader."""

import os
import tempfile
import unittest

from PIL import Image

from nn_imgLoader import nn_imgLoader


def _make_img(path, size=(8, 8), color=(120, 60, 30)):
    Image.new("RGB", size, color).save(path)


class TestNnImgLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        # Struktur: root/classA/{1,2}.jpg  root/classB/3.jpg
        self.class_a = os.path.join(self.root, "classA")
        self.class_b = os.path.join(self.root, "classB")
        os.makedirs(self.class_a)
        os.makedirs(self.class_b)
        self.img1 = os.path.join(self.class_a, "1.jpg")
        self.img2 = os.path.join(self.class_a, "2.jpg")
        self.img3 = os.path.join(self.class_b, "3.jpg")
        for p in (self.img1, self.img2, self.img3):
            _make_img(p)
        self.loader = nn_imgLoader()

    def tearDown(self):
        self.tmp.cleanup()

    def test_loadSingle_valid(self):
        obj = self.loader.loadSingle(self.img1)
        self.assertIsInstance(obj, dict)
        self.assertEqual(obj["size"], (8, 8))
        self.assertEqual(obj["format"], "JPEG")
        self.assertEqual(self.loader.statusCode, 0)

    def test_loadSingle_empty_path(self):
        result = self.loader.loadSingle("")
        self.assertIsNone(result)
        self.assertEqual(self.loader.statusCode, 1)

    def test_loadSingle_missing_file(self):
        result = self.loader.loadSingle(os.path.join(self.root, "nope.jpg"))
        self.assertIsNone(result)
        self.assertEqual(self.loader.statusCode, 1)

    def test_loadFolder_non_recursive(self):
        images = self.loader.loadFolder(self.class_a, recursive=False)
        self.assertEqual(len(images), 2)
        self.assertTrue(all(isinstance(i, dict) for i in images))

    def test_loadFolder_recursive(self):
        images = self.loader.loadFolder(self.root, recursive=True)
        self.assertEqual(len(images), 3)

    def test_loadArrayBatched(self):
        images = self.loader.loadArrayBatched(self.class_a, ["1.jpg", "2.jpg"])
        self.assertEqual(len(images), 2)

    def test_loadArrayBatched_skips_missing(self):
        images = self.loader.loadArrayBatched(self.class_a, ["1.jpg", "missing.jpg"])
        self.assertEqual(len(images), 1)

    def test_extractClassList(self):
        classes = sorted(self.loader.extractClassList(self.root))
        self.assertEqual(classes, ["classA", "classB"])

    def test_list_recursive_walk(self):
        files = self.loader.list_recursive_walk(self.root)
        self.assertEqual(len(files), 3)
        self.assertTrue(all(os.path.isfile(f) for f in files))


if __name__ == "__main__":
    unittest.main()
