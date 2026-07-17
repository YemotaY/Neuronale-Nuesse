# Runtime / Inferenz

Lädt ein trainiertes `*.h5`-Modell und klassifiziert Nusssorten auf
Einzelbildern, Videodateien oder einem Live-Kamera-Stream.

## Verwendung

```bash
# Einzelbild
python src/runtime/nn_runtime.py --source image \
    --model src/training/model/nuts_48.0.h5 \
    --input pfad/zum/bild.jpg

# Videodatei
python src/runtime/nn_runtime.py --source video \
    --model src/training/model/nuts_48.0.h5 \
    --input pfad/zum/video.mp4

# Live-Kamera (Standardgerät 0)
python src/runtime/nn_runtime.py --source camera --device 0
```

## Argumente

| Argument      | Beschreibung                                            | Standard                          |
|---------------|--------------------------------------------------------|-----------------------------------|
| `--source`    | `image`, `video` oder `camera` (erforderlich)          | –                                 |
| `--model`     | Pfad zum Keras-Modell (`*.h5`)                         | `src/training/model/nuts_48.0.h5` |
| `--input`     | Pfad zu Bild/Video (bei `image`/`video` erforderlich)  | –                                 |
| `--device`    | Kamera-Geräteindex (bei `camera`)                      | `0`                               |
| `--top-k`     | Anzahl angezeigter Top-Vorhersagen                     | `3`                               |
| `--no-window` | Bei `image` kein Fenster öffnen (nur Konsolenausgabe)  | aus                               |

## Hinweise

- Die Klassennamen werden alphabetisch aus `src/training/train/` abgeleitet –
  identisch zur Reihenfolge, die Keras beim Training verwendet.
  Stimmt die Anzahl nicht mit den Modellausgängen überein, werden nummerierte
  Labels verwendet.
- Die Vorverarbeitung (Größe `200×200`, RGB, keine Normalisierung) entspricht
  exakt dem Training in `nn_Trainer.py`.
- Fenster schließen: bei Bildern beliebige Taste, bei Video/Kamera `q`.

## Programmatische Nutzung

```python
from runtime import NutClassifier

clf = NutClassifier("src/training/model/nuts_48.0.h5")
preds = clf.predict(cv2_bgr_frame, top_k=3)   # [(label, wahrscheinlichkeit), ...]
```
