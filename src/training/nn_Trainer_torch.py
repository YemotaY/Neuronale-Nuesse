# -*- coding: utf-8 -*-
#
#  nn_Trainer_torch.py
#
#  Created by YemotaY on 2026-07-17.
#  MIT Licensed, 2026. All rights reserved.
#
#   b59190308@gmail.com
#
# pylint: disable-msg=F0401

"""PyTorch-Variante der Trainingspipeline (EfficientNet-B3, Transfer Learning).

Spiegelt die öffentliche Schnittstelle von ``nn_Trainer`` (Keras) wider, nutzt
aber ``torch`` / ``torchvision``:

  loadLearnData()   -> DataFrames + Klassenübersicht + Bildstatistik
  balance(...)      -> Balancing unterrepräsentierter Klassen per Augmentation
  configureModell() -> Datasets + DataLoader
  modellMixer()     -> EfficientNet-B3 + Klassifikationskopf
  startTraining()   -> Trainings-/Validierungsschleife (interaktiv steuerbar)
  analyzeTraining() -> Loss-/Accuracy-Plots
  simplePredict()   -> Testauswertung: Confusion-Matrix + Classification-Report
  storeModell(e,t)  -> Export als *.pt + *.labels.json

Abhängigkeiten:
torch, torchvision, pandas, numpy, matplotlib, seaborn, scikit-learn, pillow
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn.metrics
import shutil
import json
import time
import os

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# Funktioniert sowohl als Skript als auch als Paket-Import.
try:
    from .nn_imgLoader import nn_imgLoader
except ImportError:  # Ausführung als eigenständiges Skript
    from nn_imgLoader import nn_imgLoader

sns.set_style('darkgrid')

# Verzeichnis dieses Moduls (.../src/training) – unabhängig vom Arbeitsverzeichnis.
_TRAIN_ROOT = os.path.dirname(os.path.abspath(__file__))

# ImageNet-Normalisierung (torchvision-Konvention für vortrainierte Modelle).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class NutDataset(Dataset):
    """Bild-Datensatz auf Basis eines DataFrames mit ``filepaths``/``labels``."""

    def __init__(self, df, class_to_idx, transform):
        self.filepaths = df['filepaths'].tolist()
        self.labels = [class_to_idx[label] for label in df['labels'].tolist()]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        img = Image.open(self.filepaths[index]).convert('RGB')
        return self.transform(img), self.labels[index]


class nn_Trainer_torch:
    """PyTorch-Trainingspipeline für die Nuss-Klassifikation.

    Attribute:
        train_path/test_path/valid_path: Splitt-Ordnernamen unter _TRAIN_ROOT.
        imageSize: Eingabegröße (H, W), muss zur Runtime passen.
        device: 'cuda' falls verfügbar, sonst 'cpu'.
        model: das aufgebaute EfficientNet-B3.
        classes / class_count: Klassennamen (alphabetisch) und deren Anzahl.
    """

    def __init__(self):
        self.inited = True
        self.train_path = r'train'
        self.test_path = r'test'
        self.valid_path = r'valid'
        self.imgLoader = nn_imgLoader()
        self.imageSize = (200, 200)
        self.batch_size = 20
        self.epochs = 40
        self.ask_epoch = 40
        self.train_df = None
        self.test_df = None
        self.valid_df = None
        self.model = None
        self.classes = None
        self.class_count = None
        self.class_to_idx = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------- Daten laden
    def loadLearnData(self):
        for path in [self.train_path, self.test_path, self.valid_path]:
            filepaths = []
            labels = []
            classlist = self.imgLoader.extractClassList(os.path.join(_TRAIN_ROOT, path))
            for klass in classlist:
                classpath = os.path.join(_TRAIN_ROOT, path, klass)
                for f in os.listdir(classpath):
                    filepaths.append(os.path.join(classpath, f))
                    labels.append(klass)

            Fseries = pd.Series(filepaths, name='filepaths')
            Lseries = pd.Series(labels, name='labels')
            df = pd.concat([Fseries, Lseries], axis=1)

            if path == self.train_path:
                self.train_df = df
            elif path == self.test_path:
                self.test_df = df
            else:
                self.valid_df = df

        print('train_df length: ', len(self.train_df), '  test_df length: ',
              len(self.test_df), '  valid_df length: ', len(self.valid_df))

        self.classes = sorted(list(self.train_df['labels'].unique()))
        self.class_count = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print('The number of classes in the dataset is: ', self.class_count)

        groups = self.train_df.groupby('labels')
        print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
        countlist = []
        classlist = []
        for label in self.classes:
            group = groups.get_group(label)
            countlist.append(len(group))
            classlist.append(label)
            print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

        max_value = np.max(countlist)
        max_class = classlist[countlist.index(max_value)]
        min_value = np.min(countlist)
        min_class = classlist[countlist.index(min_value)]
        print(max_class, ' Hat die meisten Bilder = ', max_value, ' ',
              min_class, ' Hat die wenigsten Bilder = ', min_value)

        ht = 0
        wt = 0
        sample_size = min(100, len(self.train_df))
        train_df_sample = self.train_df.sample(n=sample_size, random_state=123, axis=0)
        for i in range(len(train_df_sample)):
            fpath = train_df_sample['filepaths'].iloc[i]
            img = plt.imread(fpath)
            ht += img.shape[0]
            wt += img.shape[1]
        print('Durchsch. Höhe = ', ht // sample_size, ' Durchsch. Breite = ',
              wt // sample_size, 'Verhältnis = ', ht / wt)

    # -------------------------------------------------------------- Balancing
    def balance(self, df, n, working_dir):
        df = df.copy()
        print('Anfangslänge des Dataframes : ', len(df))
        aug_dir = os.path.join(_TRAIN_ROOT, working_dir)

        if os.path.isdir(aug_dir):
            shutil.rmtree(aug_dir)
        os.mkdir(aug_dir)

        for label in df['labels'].unique():
            os.mkdir(os.path.join(aug_dir, label))

        # Augmentierungen analog zum Keras-ImageDataGenerator.
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.Resize(self.imageSize),
        ])

        total = 0
        groups = df.groupby('labels')
        for label in df['labels'].unique():
            group = groups.get_group(label)
            sample_count = len(group)
            if sample_count < n:
                delta = n - sample_count
                target_dir = os.path.join(aug_dir, label)
                src_paths = group['filepaths'].tolist()
                for i in range(delta):
                    src = src_paths[i % len(src_paths)]
                    try:
                        img = Image.open(src).convert('RGB')
                        aug_transform(img).save(
                            os.path.join(target_dir, f'aug-{i}.jpg'), format='JPEG')
                        total += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f'Augmentierung fehlgeschlagen für {src}: {exc}')
                print('{0:40s} für die klasse {1:^30s} erstellt {2:^5s} bearbeitete Bilder'
                      .format(' ', label, str(delta)), '\r', end='')
        print('\nGesamtzahl bearbeiteter Bilder = ', total)

        aug_fpaths = []
        aug_labels = []
        for klass in os.listdir(aug_dir):
            classpath = os.path.join(aug_dir, klass)
            for f in os.listdir(classpath):
                aug_fpaths.append(os.path.join(classpath, f))
                aug_labels.append(klass)

        aug_df = pd.concat([pd.Series(aug_fpaths, name='filepaths'),
                            pd.Series(aug_labels, name='labels')], axis=1)
        df = pd.concat([df, aug_df], axis=0).reset_index(drop=True)
        print('Die Länge des Dataframes ist nun :', len(df))
        self.train_df = df
        return df

    # -------------------------------------------------------- DataLoader-Setup
    def configureModell(self):
        train_tf = transforms.Compose([
            transforms.Resize(self.imageSize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(self.imageSize),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        self.train_ds = NutDataset(self.train_df, self.class_to_idx, train_tf)
        self.valid_ds = NutDataset(self.valid_df, self.class_to_idx, eval_tf)
        self.test_ds = NutDataset(self.test_df, self.class_to_idx, eval_tf)

        self.train_loader = DataLoader(self.train_ds, batch_size=self.batch_size,
                                       shuffle=True, num_workers=2)
        self.valid_loader = DataLoader(self.valid_ds, batch_size=self.batch_size,
                                       shuffle=False, num_workers=2)
        self.test_loader = DataLoader(self.test_ds, batch_size=self.batch_size,
                                      shuffle=False, num_workers=2)
        print('train batches: ', len(self.train_loader), ' valid batches: ',
              len(self.valid_loader), ' test batches: ', len(self.test_loader),
              ' number of classes : ', self.class_count)

    # --------------------------------------------------------- Modell aufbauen
    def modellMixer(self):
        base_model = torchvision.models.efficientnet_b3(
            weights=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, self.class_count),
        )
        self.model = base_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.001)

    # ------------------------------------------------------------- Training
    def _run_epoch(self, loader, train):
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        count = 0
        torch.set_grad_enabled(train)
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if train:
                self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            if train:
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            count += inputs.size(0)
        torch.set_grad_enabled(True)
        return total_loss / max(count, 1), correct / max(count, 1)

    def startTraining(self):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        best_vloss = np.inf
        best_state = None
        best_epoch = 1
        start_time = time.time()
        ask_epoch = self.ask_epoch

        print('Training will proceed until epoch', ask_epoch,
              '- then enter H to halt or an integer for more epochs.')

        epoch = 0
        while epoch < self.epochs:
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            v_loss, v_acc = self._run_epoch(self.valid_loader, train=False)
            history['loss'].append(tr_loss)
            history['accuracy'].append(tr_acc)
            history['val_loss'].append(v_loss)
            history['val_accuracy'].append(v_acc)
            print(f'Epoch {epoch + 1:3d}/{self.epochs}  '
                  f'loss={tr_loss:7.4f} acc={tr_acc:6.4f}  '
                  f'val_loss={v_loss:7.4f} val_acc={v_acc:6.4f}')

            if v_loss < best_vloss:
                best_vloss = v_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                print(f' validation loss {v_loss:7.4f} is a new best (epoch {best_epoch}).')

            if epoch + 1 == ask_epoch and self.epochs > 1:
                print('\n Enter H to end training or an integer for additional epochs:')
                ans = input()
                if ans in ('H', 'h', '0'):
                    print(f'Training halted on epoch {epoch + 1} due to user input.')
                    break
                if ans.isdigit():
                    ask_epoch += int(ans)
                    print(f'current LR is {self.optimizer.param_groups[0]["lr"]:.5f} '
                          f'- hit enter to keep or enter a new LR')
                    lr_ans = input(' ')
                    if lr_ans.strip():
                        new_lr = float(lr_ans)
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = new_lr
                        print(' changing LR to ', new_lr)
            epoch += 1

        if best_state is not None:
            print('loading model with weights from epoch ', best_epoch)
            self.model.load_state_dict(best_state)

        dur = time.time() - start_time
        h = dur // 3600
        m = (dur - h * 3600) // 60
        s = dur - (h * 3600 + m * 60)
        print(f'training elapsed time was {h} hours, {m:4.1f} minutes, {s:4.2f} seconds')
        return history

    # ------------------------------------------------------------- Auswertung
    def analyzeTraining(self, tr_data, start_epoch):
        tacc = tr_data['accuracy']
        tloss = tr_data['loss']
        vacc = tr_data['val_accuracy']
        vloss = tr_data['val_loss']
        Epochs = list(range(start_epoch + 1, len(tacc) + start_epoch + 1))
        index_loss = int(np.argmin(vloss))
        index_acc = int(np.argmax(vacc))
        plt.style.use('fivethirtyeight')
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        axes[0].plot(Epochs, tloss, 'r', label='Training loss')
        axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
        axes[0].scatter(index_loss + 1 + start_epoch, vloss[index_loss], s=150, c='blue',
                        label='best epoch= ' + str(index_loss + 1 + start_epoch))
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
        axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
        axes[1].scatter(index_acc + 1 + start_epoch, vacc[index_acc], s=150, c='blue',
                        label='best epoch= ' + str(index_acc + 1 + start_epoch))
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------- Vorhersage
    def simplePredict(self):
        self.model.eval()
        y_pred = []
        y_true = []
        errors = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(targets.numpy().tolist())
        errors = int(np.sum(np.array(y_pred) != np.array(y_true)))
        tests = len(y_true)
        acc = (1 - errors / tests) * 100
        print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}')

        if self.class_count <= 30:
            cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
            plt.xticks(np.arange(self.class_count) + .5, self.classes, rotation=90)
            plt.yticks(np.arange(self.class_count) + .5, self.classes, rotation=0)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.show()
        clr = sklearn.metrics.classification_report(
            y_true, y_pred, target_names=self.classes, digits=4)
        print("Classification Report:\n----------------------\n", clr)
        return errors, tests

    # --------------------------------------------------------- Modell sichern
    def storeModell(self, errors, tests):
        subject = 'nuts'
        acc = str((1 - errors / tests) * 100)
        acc = acc[:acc.rfind('.') + 3]
        model_dir = os.path.join(_TRAIN_ROOT, "model")
        os.makedirs(model_dir, exist_ok=True)
        model_save_loc = os.path.join(model_dir, f'{subject}_{acc}.pt')
        torch.save({'state_dict': self.model.state_dict(),
                    'classes': self.classes,
                    'image_size': self.imageSize}, model_save_loc)
        print('model was saved as ', model_save_loc)

        if self.classes:
            labels_loc = os.path.join(model_dir, f'{subject}_{acc}.labels.json')
            with open(labels_loc, 'w', encoding='utf-8') as fh:
                json.dump(self.classes, fh, ensure_ascii=False, indent=2)
            print('labels were saved as ', labels_loc)


if __name__ == "__main__":
    print("================================= 0. nn_Trainer_torch: Initiere ... ===========================")
    trainer = nn_Trainer_torch()
    print(f"   Device: {trainer.device}")
    print("================================= 1. nn_Trainer_torch: Lade Lerndaten ... =====================")
    trainer.loadLearnData()
    print("================================= 2. nn_Trainer_torch: Lerndaten ausbalancieren ... ===========")
    trainer.balance(trainer.train_df, 200, r'aug')
    print("================================= 3. nn_Trainer_torch: DataLoader konfigurieren ... ===========")
    trainer.configureModell()
    print("================================= 4. nn_Trainer_torch: Basismodell einmixen ... ===============")
    trainer.modellMixer()
    print("================================= 5. nn_Trainer_torch: Starte das Training ... =================")
    history = trainer.startTraining()
    print("================================= 6. nn_Trainer_torch: Auswertung ... ==========================")
    trainer.analyzeTraining(history, 0)
    print("================================= 7. nn_Trainer_torch: Vorhersage ... ==========================")
    errors, tests = trainer.simplePredict()
    print("================================= 8. nn_Trainer_torch: Speichere das Modell ... ================")
    trainer.storeModell(errors, tests)
    print("================================= nn_Trainer_torch: ALLES FERTIG ===============================")
