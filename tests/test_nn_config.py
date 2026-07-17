# -*- coding: utf-8 -*-
"""Unit-Tests für die Config-Verwaltung (nn_config)."""

import json
import os
import tempfile
import unittest

from nn_config import TrainerConfig, load_config, save_config


class TestNnConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TrainerConfig()
        self.assertEqual(cfg.image_size, (200, 200))
        self.assertEqual(cfg.batch_size, 20)
        self.assertEqual(cfg.epochs, 40)

    def test_load_missing_file_returns_defaults(self):
        cfg = load_config(os.path.join(tempfile.gettempdir(), "does_not_exist_xyz.json"))
        self.assertEqual(cfg.batch_size, 20)

    def test_roundtrip_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"image_size": [64, 64], "batch_size": 4,
                           "epochs": 2, "unknown_key": 99}, fh)
            cfg = load_config(path)
            self.assertEqual(cfg.image_size, (64, 64))
            self.assertEqual(cfg.batch_size, 4)
            self.assertEqual(cfg.epochs, 2)
            # unbekannte Schlüssel werden ignoriert, Rest bleibt Default
            self.assertEqual(cfg.balance_count, 200)

    def test_save_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            save_config(TrainerConfig(batch_size=8), path)
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["batch_size"], 8)
            self.assertEqual(data["image_size"], [200, 200])


if __name__ == "__main__":
    unittest.main()
