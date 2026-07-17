# Neuronale Nüsse 2.0

Bildklassifikation von Nusssorten mit einem auf **EfficientNetB3** basierenden
Transfer-Learning-Modell (TensorFlow / Keras). Das Projekt umfasst den kompletten
Weg von der Datenaufbereitung über das interaktive Training bis hin zur
Laufzeit-Erkennung (Bild, Video, Kamera-Stream).

Aktuell werden **10 Nussklassen** unterschieden:

`cashewkern`, `Haselnuss`, `Kokosnuss`, `makadamia`, `mandeln`, `paranuss`,
`pekanüsse`, `pinienkerne`, `pistazien`, `wallnuss`

---

## Inhaltsverzeichnis

- [Features](#features)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Datensatz vorbereiten](#datensatz-vorbereiten)
- [Training](#training)
- [Bild-Tagger (GUI)](#bild-tagger-gui)
- [Runtime / Inferenz](#runtime--inferenz)
- [Ergebnisse](#ergebnisse)
- [Roadmap](#roadmap)
- [Lizenz](#lizenz)

---

## Features

- **Transfer Learning** auf Basis von EfficientNetB3 (ImageNet-Gewichte).
- **Datenausgleich** (Balancing) unterrepräsentierter Klassen über
  Data-Augmentation (`ImageDataGenerator`).
- **Interaktives Training**: über einen Keras-Callback kann das Training zur
  Laufzeit angehalten, verlängert und die Lernrate angepasst werden.
- **Automatisches Speichern der besten Gewichte** (niedrigster Validierungs-Loss).
- **Auswertung** mit Loss-/Accuracy-Plots, Confusion-Matrix und
  Classification-Report.
- **Bild-Tagger-GUI** (Tkinter) zum Verschlagworten von Bildern über EXIF-Metadaten.
- **Runtime-Modul** für die Erkennung auf Einzelbildern, Videos und Live-Kamera.

---

## Projektstruktur

```
Neuronale Nüsse 2.0/
├── readme.md
├── requirements.txt
├── results/                       # Auswertungs-Plots (Confusion-Matrix, Verlauf)
└── src/
    ├── docs/                       # Projektdokumentation & Präsentation (DE)
    ├── legacy_old/                 # Alte Notebooks (Referenz / Archiv)
    ├── runtime/                    # Inferenz zur Laufzeit (Bild / Video / Kamera)
    └── training/
        ├── nn_Trainer.py            # Haupt-Trainingspipeline
        ├── nn_imgLoader.py          # Laden von Bildern + Metadaten
        ├── nn_interactiveTraining.py# Keras-Callback für interaktives Training
        ├── nn_gui_imgTagger.py      # Tkinter-GUI zum EXIF-Tagging
        ├── model/                   # Trainierte Modelle (*.h5)
        ├── train/  valid/  test/    # Datensplits (je Klasse ein Unterordner)
        └── aug/                      # Generierte Augmentierungs-Bilder
```

---

## Installation

Voraussetzung: **Python 3.11+**.

```bash
# Repository klonen und in das Projektverzeichnis wechseln
cd "Neuronale Nüsse 2.0"

# Virtuelle Umgebung anlegen
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

Kernabhängigkeiten: `tensorflow`, `keras`, `opencv-python`, `pillow`,
`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

---

## Datensatz vorbereiten

Die Bilder werden nach folgendem Schema erwartet – je Split ein Ordner, darin je
Klasse ein Unterordner:

```
src/training/
├── train/<klasse>/*.jpg
├── valid/<klasse>/*.jpg
└── test/<klasse>/*.jpg
```

Die Klassennamen ergeben sich automatisch aus den Ordnernamen. Neue Klassen
lassen sich hinzufügen, indem einfach ein weiterer Ordner in allen drei Splits
angelegt wird.

---

## Training

Die gesamte Pipeline wird über `nn_Trainer.py` gestartet:

```bash
cd "Neuronale Nüsse 2.0"
python src/training/nn_Trainer.py
```

Ablauf der Pipeline:

1. **Lerndaten laden** – Dateipfade und Labels in DataFrames überführen.
2. **Balancing** – unterrepräsentierte Klassen per Augmentation auf ein
   Zielminimum anheben.
3. **Generatoren konfigurieren** – Train/Valid/Test-`ImageDataGenerator`.
4. **Modell zusammenbauen** – EfficientNetB3 + Dense-Kopf.
5. **Training** – interaktiv über den Callback steuerbar.
6. **Auswertung** – Loss/Accuracy-Plots.
7. **Vorhersage & Speichern** – Confusion-Matrix, Report, `*.h5`-Export.

Während des Trainings kann bei Erreichen der `ask_epoch`:

- `H` eingegeben werden, um das Training zu beenden,
- eine Zahl eingegeben werden, um weitere Epochen zu trainieren,
- optional eine neue Lernrate gesetzt werden.

Das Modell wird unter `src/training/model/nuts_<accuracy>.h5` gespeichert.
Zusätzlich werden die Klassennamen als `nuts_<accuracy>.labels.json` abgelegt,
damit die Runtime die Label-Reihenfolge nicht rekonstruieren muss.

Ausführliche Doku: [src/training/README.md](src/training/README.md).

---

## Bild-Tagger (GUI)

Zum Verschlagworten von Bildern steht eine Tkinter-Oberfläche bereit. Tags
werden in die EXIF-Metadaten (`XPKeywords` + `ImageDescription`) geschrieben.

```bash
python src/training/nn_gui_imgTagger.py
```

- **Links:** ein-/ausklappbare, scrollbare Ordnerübersicht (`src/img`).
- **Rechts:** Vorschau, Tag-Eingabefeld und Speichern.

---

## Runtime / Inferenz

Das Runtime-Modul lädt ein trainiertes `*.h5`-Modell und klassifiziert
Einzelbilder, Videodateien oder einen Live-Kamera-Stream.

```bash
# Einzelbild
python src/runtime/nn_runtime.py --model src/training/model/nuts_48.0.h5 \
    --source image --input pfad/zum/bild.jpg

# Videodatei
python src/runtime/nn_runtime.py --model src/training/model/nuts_48.0.h5 \
    --source video --input pfad/zum/video.mp4

# Live-Kamera (Standardgerät 0)
python src/runtime/nn_runtime.py --model src/training/model/nuts_48.0.h5 \
    --source camera --device 0
```

Details siehe [src/runtime/README.md](src/runtime/README.md).

---

## Ergebnisse

Auswertungen des letzten Trainingslaufs liegen unter `results/`:

- `Training_30_epochs.png` – Loss-/Accuracy-Verlauf über 30 Epochen.
- `ConfusionMatrix_30_epochs.png` – Confusion-Matrix auf dem Testset.

---

## Dokumentation

Die ausführliche Projektdokumentation und die Präsentation liegen unter
`src/docs/` (jeweils als PDF und im OpenDocument-Original).

<details>
<summary>📄 Projektdokumentation (PDF)</summary>

- [Projektdoku_DE.pdf](src/docs/Projektdoku_DE.pdf) – ausführliche Projektdokumentation
- [Projektdoku_DE.odt](src/docs/Projektdoku_DE.odt) – OpenDocument-Original

> Hinweis: GitHub bindet PDFs nicht inline ein. Der Link öffnet die Datei im
> PDF-Viewer bzw. lädt sie herunter.

</details>

<details>
<summary>📊 Präsentation (PDF)</summary>

- [Neuronale Nüsse_Praesentation_DE.pdf](src/docs/Neuronale%20N%C3%BCsse_Praesentation_DE.pdf) – Projektpräsentation

</details>

---

## Roadmap

- [x] Klassennamen zusammen mit dem Modell serialisieren (`*.labels.json`).
- [ ] Konfiguration (Bildgröße, Batch-Size, Epochen) in eine Config-Datei auslagern.
- [ ] Umstieg von `ImageDataGenerator` auf `tf.data` / `keras.utils.image_dataset_from_directory`.
- [ ] Unit-Tests für Loader und Runtime.
- [ ] Export nach TensorFlow Lite für Edge-Geräte.

---

## Lizenz

MIT Licensed, 2026. Siehe Kopfzeilen der Quelldateien.
Kontakt: b59190308@gmail.com
