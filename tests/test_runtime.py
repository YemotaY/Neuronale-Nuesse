# -*- coding: utf-8 -*-
"""Unit-Tests für das Runtime-Modul (nn_runtime).

Baut ein winziges echtes Keras-Modell, um die tatsächliche Lade-/Inferenz-
Pipeline (inkl. tf.keras.models.load_model) zu testen – ohne EfficientNet.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import nn_runtime
from nn_runtime import NutClassifier, run_image


def _build_tiny_model(path, n_classes, image_size=(200, 200)):
    inp = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inp)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.save(path)
    return model


class TestNutClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_path = os.path.join(cls.tmp.name, "tiny.h5")
        cls.labels = ["alpha", "beta", "gamma"]
        _build_tiny_model(cls.model_path, len(cls.labels))
        # labels.json neben das Modell legen
        with open(os.path.splitext(cls.model_path)[0] + ".labels.json", "w",
                  encoding="utf-8") as fh:
            json.dump(cls.labels, fh)
        cls.clf = NutClassifier(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_class_names_from_labels_json(self):
        self.assertEqual(self.clf.class_names, self.labels)

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            NutClassifier(os.path.join(self.tmp.name, "nope.h5"))

    def test_preprocess_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        batch = self.clf.preprocess(frame)
        self.assertEqual(batch.shape, (1, 200, 200, 3))
        self.assertEqual(batch.dtype, np.float32)

    def test_predict_returns_sorted_topk(self):
        frame = (np.random.rand(120, 120, 3) * 255).astype(np.uint8)
        preds = self.clf.predict(frame, top_k=2)
        self.assertEqual(len(preds), 2)
        # Labels gültig
        for label, prob in preds:
            self.assertIn(label, self.labels)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        # absteigend sortiert
        self.assertGreaterEqual(preds[0][1], preds[1][1])

    def test_predict_topk_capped_to_class_count(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        preds = self.clf.predict(frame, top_k=99)
        self.assertEqual(len(preds), len(self.labels))

    def test_draw_predictions_keeps_shape(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        out = nn_runtime._draw_predictions(frame, [("alpha", 0.9)])
        self.assertEqual(out.shape, (200, 200, 3))

    def test_run_image_no_window(self):
        img_path = os.path.join(self.tmp.name, "sample.jpg")
        import cv2
        cv2.imwrite(img_path, np.zeros((50, 50, 3), dtype=np.uint8))
        preds = run_image(self.clf, img_path, top_k=3, show=False)
        self.assertEqual(len(preds), 3)

    def test_run_image_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            run_image(self.clf, os.path.join(self.tmp.name, "missing.jpg"), show=False)


if __name__ == "__main__":
    unittest.main()
