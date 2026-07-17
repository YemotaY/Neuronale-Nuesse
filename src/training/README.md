# Training

Trainingspipeline für das Nuss-Klassifikationsmodell (EfficientNetB3,
Transfer Learning). Dieses Paket deckt Datenaufbereitung, Balancing, Training,
Auswertung und Modell-Export ab und enthält zusätzlich eine GUI zum Tagging.

## Inhalt

| Datei                        | Zweck                                                                 |
|------------------------------|-----------------------------------------------------------------------|
| `nn_Trainer.py`              | Haupt-Trainingspipeline (Keras/TensorFlow, per `__main__` ausführbar). |
| `nn_Trainer_torch.py`        | Gleiche Pipeline als **PyTorch**-Variante (EfficientNet-B3).           |
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
└── model/                   # exportierte Modelle (*.h5) + *.labels.json
```

## Schnellstart

```bash
cd "Neuronale Nüsse 2.0"
python src/training/nn_Trainer.py
```

Die Pipeline durchläuft: Laden → Balancing → Generatoren → Modellaufbau →
Training → Auswertung → Vorhersage → Speichern.

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
| `configureModell()`  | Erstellt die `ImageDataGenerator` und berechnet die Test-Batch-Größe.        |
| `modellMixer()`      | Baut EfficientNetB3 + Dense-Kopf und kompiliert das Modell.                   |
| `startTraining()`    | Startet `model.fit` mit dem interaktiven Callback.                            |
| `analyzeTraining()`  | Plottet Loss-/Accuracy-Verläufe inkl. bester Epoche.                          |
| `simplePredict()`    | Vorhersage auf dem Testset, Confusion-Matrix + Classification-Report.        |
| `storeModell(e, t)`  | Speichert Modell als `nuts_<acc>.h5` **und** die Klassen als `.labels.json`. |

Zentrale Parameter (in `nn_Trainer.__init__` / Methoden):

- `imageSize = (200, 200)` – Eingabegröße (muss mit der Runtime übereinstimmen).
- `batch_size = 20` – Training/Validierung.
- `epochs = 40`, `ask_epoch = 40` – Trainingsdauer und Abfrageintervall.
- Balance-Ziel `n = 200` (im `__main__`-Aufruf gesetzt).

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
