# -*- coding: utf-8 -*-
#
#  nn_runtime.py
#
#  Created by YemotaY on 2026-07-17.
#  MIT Licensed, 2026. All rights reserved.
#
#   b59190308@gmail.com
#
# pylint: disable-msg=F0401

"""Laufzeit-Inferenz für das Nuss-Klassifikationsmodell.

Lädt ein trainiertes Keras-Modell (*.h5) und klassifiziert:
  - Einzelbilder  (--source image)
  - Videodateien  (--source video)
  - Live-Kamera    (--source camera)

Die Vorverarbeitung entspricht dem Training (nn_Trainer):
Bilder werden auf IMAGE_SIZE skaliert und als RGB ohne zusätzliche
Normalisierung an das Modell gegeben (EfficientNet enthält die
Skalierung intern).

Abhängigkeiten:
tensorflow, opencv-python, numpy
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

# Muss mit dem Training übereinstimmen (nn_Trainer.imageSize)
IMAGE_SIZE = (200, 200)

# Standardpfade relativ zum Projekt-Root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
_DEFAULT_MODEL = os.path.join(_PROJECT_ROOT, "src", "training", "model", "nuts_48.0.h5")
_DEFAULT_TRAIN_DIR = os.path.join(_PROJECT_ROOT, "src", "training", "train")


class NutClassifier:
    """Kapselt Modell, Klassennamen und die Inferenz-Vorverarbeitung."""

    def __init__(self, model_path, class_names=None, image_size=IMAGE_SIZE):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        # Eingabegröße bevorzugt aus dem Modell ableiten (robust gegen
        # abweichende Defaults); sonst den übergebenen Wert nutzen.
        self.image_size = self._infer_image_size(image_size)
        self.class_names = class_names or self._infer_class_names()

        out_units = int(self.model.output_shape[-1])
        if len(self.class_names) != out_units:
            print(
                f"WARNUNG: {len(self.class_names)} Klassennamen, aber das Modell "
                f"hat {out_units} Ausgänge. Nummerierte Labels werden verwendet.",
                file=sys.stderr,
            )
            self.class_names = [f"Klasse_{i}" for i in range(out_units)]

    def _infer_image_size(self, fallback):
        """Liest (H, W) aus der Modell-Eingabeform, falls verfügbar."""
        try:
            shape = self.model.input_shape  # z. B. (None, 200, 200, 3)
            if isinstance(shape, list):
                shape = shape[0]
            if shape and shape[1] and shape[2]:
                return (int(shape[1]), int(shape[2]))
        except (AttributeError, IndexError, TypeError):
            pass
        return fallback

    def _infer_class_names(self):
        """Leitet die Klassennamen ab.

        1. Bevorzugt eine ``<modell>.labels.json`` neben dem Modell
           (wird von ``nn_Trainer.storeModell`` geschrieben).
        2. Fallback: alphabetische Ordnernamen aus ``src/training/train``
           \u2013 das entspricht der Reihenfolge von ``class_indices`` in Keras.
        """
        labels_path = os.path.splitext(self.model_path)[0] + ".labels.json"
        if os.path.isfile(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as fh:
                    names = json.load(fh)
                if isinstance(names, list) and names:
                    return [str(n) for n in names]
            except (OSError, ValueError) as exc:
                print(f"WARNUNG: labels.json nicht lesbar ({exc}).", file=sys.stderr)

        if os.path.isdir(_DEFAULT_TRAIN_DIR):
            names = sorted(
                d for d in os.listdir(_DEFAULT_TRAIN_DIR)
                if os.path.isdir(os.path.join(_DEFAULT_TRAIN_DIR, d))
            )
            if names:
                return names
        return []

    def preprocess(self, frame_bgr):
        """Wandelt ein OpenCV-BGR-Frame in einen Modell-Batch (1, H, W, 3)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized.astype(np.float32), axis=0)

    def predict(self, frame_bgr, top_k=3):
        """Gibt die Top-k ``(label, wahrscheinlichkeit)`` für ein Frame zurück."""
        batch = self.preprocess(frame_bgr)
        probs = self.model.predict(batch, verbose=0)[0]
        k = min(top_k, len(probs))
        top_idx = np.argsort(probs)[::-1][:k]
        return [(self._label(i), float(probs[i])) for i in top_idx]

    def _label(self, index):
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return f"Klasse_{index}"


# ------------------------------------------------------------------ Overlay
def _draw_predictions(frame, predictions):
    """Zeichnet die Top-k-Vorhersagen als Overlay in das Frame."""
    y = 28
    for label, prob in predictions:
        text = f"{label}: {prob * 100:5.1f}%"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 220, 0), 2, cv2.LINE_AA)
        y += 32
    return frame


# ------------------------------------------------------------------ Sources
def run_image(classifier, input_path, top_k=3, show=True):
    """Klassifiziert ein Einzelbild und gibt die Vorhersagen aus."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Bild nicht gefunden: {input_path}")

    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"Bild konnte nicht gelesen werden: {input_path}")

    predictions = classifier.predict(frame, top_k=top_k)
    print(f"\nVorhersage für {os.path.basename(input_path)}:")
    for label, prob in predictions:
        print(f"  {label:<15s} {prob * 100:6.2f}%")

    if show:
        _draw_predictions(frame, predictions)
        cv2.imshow("Neuronale Nuesse - Bild", frame)
        print("\nBeliebige Taste im Fenster drücken zum Beenden ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return predictions


def _run_stream(classifier, capture, top_k, window_title):
    """Gemeinsame Schleife für Video- und Kamera-Quellen."""
    if not capture.isOpened():
        raise RuntimeError("Videoquelle konnte nicht geöffnet werden.")

    print("Beenden mit 'q' im Fenster.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            predictions = classifier.predict(frame, top_k=top_k)
            _draw_predictions(frame, predictions)
            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_video(classifier, input_path, top_k=3):
    """Klassifiziert die Frames einer Videodatei."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Video nicht gefunden: {input_path}")
    capture = cv2.VideoCapture(input_path)
    _run_stream(classifier, capture, top_k, "Neuronale Nuesse - Video")


def run_camera(classifier, device=0, top_k=3):
    """Klassifiziert den Live-Stream einer Kamera."""
    capture = cv2.VideoCapture(device)
    _run_stream(classifier, capture, top_k, "Neuronale Nuesse - Kamera")


# --------------------------------------------------------------------- CLI
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Laufzeit-Inferenz für das Nuss-Klassifikationsmodell.",
    )
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="Pfad zum trainierten Keras-Modell (*.h5).")
    parser.add_argument("--source", choices=("image", "video", "camera"),
                        required=True, help="Art der Eingabequelle.")
    parser.add_argument("--input",
                        help="Pfad zu Bild/Video (bei --source image|video).")
    parser.add_argument("--device", type=int, default=0,
                        help="Kamera-Geräte-Index (bei --source camera).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Anzahl der angezeigten Top-Vorhersagen.")
    parser.add_argument("--no-window", action="store_true",
                        help="Bei --source image kein Fenster öffnen.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.source in ("image", "video") and not args.input:
        print(f"--input ist bei --source {args.source} erforderlich.", file=sys.stderr)
        return 2

    classifier = NutClassifier(args.model)
    print(f"Modell geladen: {args.model}")
    print(f"Klassen ({len(classifier.class_names)}): {', '.join(classifier.class_names)}")

    if args.source == "image":
        run_image(classifier, args.input, top_k=args.top_k, show=not args.no_window)
    elif args.source == "video":
        run_video(classifier, args.input, top_k=args.top_k)
    else:
        run_camera(classifier, device=args.device, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
