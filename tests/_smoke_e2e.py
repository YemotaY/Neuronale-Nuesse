# -*- coding: utf-8 -*-
"""Headless End-to-End-Smoke-Test der TF-Trainingspipeline (2 Epochen).

Läuft ohne GUI (Agg-Backend, plt.show wird deaktiviert) und ohne interaktive
Eingabe (ask_epoch > epochs). Benötigt Internet für die EfficientNetB3-Gewichte.

Aufruf:
    MPLBACKEND=Agg .venv/bin/python tests/_smoke_e2e.py
"""

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "training"))

import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # Fenster unterdrücken

from nn_config import TrainerConfig
from nn_Trainer import nn_Trainer


def main():
    cfg = TrainerConfig(
        image_size=(200, 200),
        batch_size=8,
        epochs=2,
        ask_epoch=3,        # > epochs -> keine interaktive Abfrage
        balance_count=8,    # klein halten für schnellen Testlauf
        learning_rate=0.001,
    )

    trainer = nn_Trainer(config=cfg)
    print(">>> loadLearnData")
    trainer.loadLearnData()
    print(">>> balance")
    trainer.balance(trainer.train_df, cfg.balance_count, trainer.aug_path)
    print(">>> configureModell")
    trainer.configureModell()
    print(">>> modellMixer")
    trainer.modellMixer()
    print(">>> startTraining (2 Epochen)")
    history = trainer.startTraining()
    print(">>> analyzeTraining")
    trainer.analyzeTraining(history, 0)
    print(">>> simplePredict")
    errors, tests = trainer.simplePredict()
    print(">>> storeModell")
    trainer.storeModell(errors, tests)
    print(">>> exportTFLite")
    tflite_path = trainer.exportTFLite(errors, tests)

    assert os.path.isfile(tflite_path), "TFLite-Datei wurde nicht erstellt"
    print("\nSMOKE-TEST OK ->", tflite_path)


if __name__ == "__main__":
    main()
