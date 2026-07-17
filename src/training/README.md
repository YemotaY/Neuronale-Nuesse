# Training

Trainingspipeline für das Nuss-Klassifikationsmodell (EfficientNetB3,
Transfer Learning). Dieses Paket deckt Datenaufbereitung, Balancing, Training,
Auswertung und Modell-Export ab und enthält zusätzlich eine GUI zum Tagging.

## Inhalt

| Datei                        | Zweck                                                                 |
|------------------------------|-----------------------------------------------------------------------|
| `nn_Trainer.py`              | Haupt-Trainingspipeline (Keras/TensorFlow, per `__main__` ausführbar). |
| `nn_Trainer_torch.py`        | Gleiche Pipeline als **PyTorch**-Variante (EfficientNet-B3).           |
| `nn_config.py`               | Zentrale Konfiguration (lädt `config.json`, Defaults als Dataclass).  |
| `config.json`                | Hyperparameter: Bildgröße, Batch-Size, Epochen, Balancing, LR, Seed.  |
| `nn_imgLoader.py`            | Laden von Bildern + EXIF-Metadaten, Klassen-/Ordner-Auflistung.       |
| `nn_interactiveTraining.py`  | Keras-Callback zum interaktiven Steuern des Trainings.                |
| `nn_gui_imgTagger.py`        | Tkinter-GUI zum Verschlagworten von Bildern (EXIF).                   |
| `__init__.py`                | Paket-Exports (`nn_Trainer`, `nn_imgLoader`, `interactiveTraining`).  |

## Verzeichnisse

```
training/
├── train/<klasse>/*.jpg     # Trainingsdaten (je Klasse ein Ordner)
├── valid/<klasse>/*.jpg     # Validierungsdaten
├── test/<klasse>/*.jpg      # Testdaten
├── aug/                     # generierte Augmentierungs-Bilder (Balancing)
└── model/                   # exportierte Modelle (*.h5 / *.tflite) + *.labels.*
```

## Schnellstart

```bash
cd "Neuronale Nüsse 2.0"
python src/training/nn_Trainer.py
```

Die Pipeline durchläuft: Laden → Balancing → Datasets → Modellaufbau →
Training → Auswertung → Vorhersage → Speichern → TFLite-Export.

## Konfiguration (`nn_config` / `config.json`)

Alle Hyperparameter liegen zentral in `config.json` und werden beim Erzeugen
des Trainers automatisch geladen. Fehlt die Datei, greifen die Defaults aus
`TrainerConfig`.

```json
{
  "image_size": [200, 200],
  "batch_size": 20,
  "epochs": 40,
  "ask_epoch": 40,
  "balance_count": 200,
  "learning_rate": 0.001,
  "seed": 123
}
```

```python
from nn_config import TrainerConfig
from nn_Trainer import nn_Trainer

# aus config.json (Default)
trainer = nn_Trainer()

# oder programmatisch überschreiben
trainer = nn_Trainer(config=TrainerConfig(epochs=2, batch_size=8))
```

Eine Vorlage schreiben: `python src/training/nn_config.py`.

## Datenpipeline (tf.data)

Statt des veralteten `ImageDataGenerator` nutzt der Trainer nun
`keras.utils.image_dataset_from_directory` + `tf.data`:

- Konsistente, alphabetische Klassenreihenfolge über alle Splits (`class_names`).
- Augmentierung on-the-fly über Keras-Preprocessing-Layer (`RandomFlip`,
  `RandomRotation`, `RandomTranslation`, `RandomZoom`) – nur auf Trainingsdaten.
- `prefetch(AUTOTUNE)` für bessere Auslastung.
- Balancing-Bilder aus `aug/` werden automatisch an das Trainings-Dataset
  angehängt.

## TensorFlow-Lite-Export (Edge-Geräte)

`exportTFLite(errors, tests, quantize=True)` konvertiert das trainierte Modell
nach `model/nuts_<acc>.tflite` (mit Dynamic-Range-Quantisierung) und schreibt
die Labels zusätzlich als `nuts_<acc>.labels.txt` für minimale Edge-Runtimes.

## Tests

Unit-Tests liegen unter `tests/` (stdlib `unittest`, kein pytest nötig):

```bash
# schnelle Tests (Loader + Config)
python -m unittest tests.test_nn_imgLoader tests.test_nn_config

# Runtime-Tests (baut ein winziges Keras-Modell)
python -m unittest tests.test_runtime

# End-to-End-Smoke-Test der Pipeline (2 Epochen, headless)
MPLBACKEND=Agg python tests/_smoke_e2e.py
```

## PyTorch-Variante (`nn_Trainer_torch`)

Funktionsgleiche Pipeline auf Basis von **PyTorch / torchvision** mit derselben
öffentlichen Schnittstelle (`loadLearnData`, `balance`, `configureModell`,
`modellMixer`, `startTraining`, `analyzeTraining`, `simplePredict`,
`storeModell`).

```bash
pip install torch torchvision           # zusätzliche Abhängigkeit
python src/training/nn_Trainer_torch.py
```

Unterschiede zur Keras-Version:

- Backbone: `torchvision.models.efficientnet_b3` (ImageNet-Gewichte).
- Datenzufuhr über `Dataset` + `DataLoader` statt `ImageDataGenerator`;
  Augmentierung via `torchvision.transforms`, ImageNet-Normalisierung.
- Explizite Trainings-/Validierungsschleife; beste Gewichte (niedrigster
  `val_loss`) werden gehalten. Interaktive Steuerung (Halten / weitere Epochen /
  neue Lernrate) ist direkt in `startTraining()` integriert.
- Nutzt automatisch die GPU (`cuda`), falls verfügbar (`trainer.device`).
- Export als `model/nuts_<acc>.pt` (State-Dict + Klassen + Bildgröße) **plus**
  `nuts_<acc>.labels.json`.

> Hinweis: `torch`/`torchvision` sind nicht in `requirements.txt` gelistet und
> müssen bei Nutzung dieser Variante separat installiert werden.

## Pipeline im Detail (`nn_Trainer`)

| Methode              | Aufgabe                                                                       |
|----------------------|-------------------------------------------------------------------------------|
| `loadLearnData()`    | Baut Train/Valid/Test-DataFrames, ermittelt Klassen und Bildstatistiken.     |
| `balance(df, n, ..)` | Hebt unterrepräsentierte Klassen per Augmentation auf mind. `n` Bilder an.    |
| `configureModell()`  | Baut Train/Valid/Test-`tf.data`-Datasets (`image_dataset_from_directory`).    |
| `modellMixer()`      | Baut EfficientNetB3 + Dense-Kopf und kompiliert das Modell.                   |
| `startTraining()`    | Startet `model.fit` mit dem interaktiven Callback.                            |
| `analyzeTraining()`  | Plottet Loss-/Accuracy-Verläufe inkl. bester Epoche.                          |
| `simplePredict()`    | Vorhersage auf dem Testset, Confusion-Matrix + Classification-Report.        |
| `storeModell(e, t)`  | Speichert Modell als `nuts_<acc>.h5` **und** die Klassen als `.labels.json`. |
| `exportTFLite(e, t)` | Exportiert ein quantisiertes `nuts_<acc>.tflite` + `.labels.txt`.            |

Zentrale Parameter kommen aus `config.json` (siehe oben) und sind über
`trainer.config` bzw. direkt als Attribute (`imageSize`, `batch_size`,
`epochs`, `ask_epoch`, `learning_rate`, `seed`) verfügbar.

## Interaktives Training (`nn_interactiveTraining`)

Der Callback sichert automatisch die Gewichte mit dem niedrigsten
Validierungs-Loss. Bei Erreichen von `ask_epoch`:

- `H` → Training beenden,
- Ganzzahl → so viele weitere Epochen trainieren,
- danach optional eine neue Lernrate setzen (Enter = beibehalten).

## Bild-Loader (`nn_imgLoader`)

| Methode                     | Zweck                                                       |
|-----------------------------|-------------------------------------------------------------|
| `loadSingle(path)`          | Lädt ein Bild inkl. Format/Größe/Modus/EXIF; `None` bei Fehler. |
| `loadArrayBatched(dir, arr)`| Lädt mehrere Bilder aus `dir` nach Namensliste.             |
| `loadFolder(dir, recursive)`| Lädt einen Ordner (optional rekursiv).                     |
| `extractClassList(dir)`     | Listet die Klassenordner.                                   |
| `list_recursive_walk(dir)`  | Sammelt rekursiv alle Dateipfade.                           |

`statusCode` spiegelt das Ergebnis der letzten `loadSingle`-Ausführung
(`0` = OK, `1` = Fehler).

## Bild-Tagger-GUI (`nn_gui_imgTagger`)

```bash
python src/training/nn_gui_imgTagger.py
```

Schreibt Tags in die EXIF-Felder `XPKeywords` (0x9C9E) und
`ImageDescription` (0x010E). Bildwurzel: `src/img`.

## Verbesserungen in dieser Version

- **Pfad-Robustheit:** alle Pfade relativ zu `__file__` (`_TRAIN_ROOT`) statt
  `os.getcwd()` – die Pipeline läuft nun unabhängig vom Arbeitsverzeichnis.
- **Bugfix `storeModell`:** `errors`/`tests` werden als Parameter übergeben
  statt aus dem globalen Namensraum gelesen; `model/` wird bei Bedarf erstellt.
- **Label-Export:** Klassennamen werden als `*.labels.json` neben dem Modell
  gespeichert; die Runtime nutzt diese Reihenfolge direkt.
- **Bugfix Bildstatistik:** Durchschnittshöhe/-breite teilt durch die echte
  Stichprobengröße statt fest durch `100`.
- **Bugfix `analyzeTraining`:** `plt.tight_layout()` wird nun tatsächlich aufgerufen.
- **Bugfix `nn_imgLoader.loadFolder`:** nicht-rekursiver Modus prüft `isfile`
  gegen den vollständigen Pfad und findet dadurch Dateien.
- **Konsistente Rückgabe:** `loadSingle` gibt bei Fehlern `None` zurück und
  setzt `statusCode`; Batch-Loader überspringen fehlgeschlagene Bilder.
- **Import-Kompatibilität:** `nn_Trainer` funktioniert als Skript *und* als
  Paket-Import (relativer Import mit Fallback).
- **Sauberes `__init__.py`** mit definierten Exports.

## Programmatische Nutzung

```python
from training import nn_Trainer, nn_imgLoader

loader = nn_imgLoader()
img = loader.loadSingle("pfad/zum/bild.jpg")

trainer = nn_Trainer()
trainer.loadLearnData()
```
