# -*- coding: utf-8 -*-
#
#  nn_Trainer.py
#
#  Created by YemotaY on 2026-07-17.
#  MIT Licensed, 2026. All rights reserved.
#
#   b59190308@gmail.com
#
# pylint: disable-msg=F0401

# Funktioniert sowohl als Skript (python nn_Trainer.py) als auch als Paket-Import.
try:
    from .nn_imgLoader import nn_imgLoader
    from .nn_interactiveTraining import interactiveTraining
    from .nn_config import load_config
except ImportError:  # Ausführung als eigenständiges Skript
    from nn_imgLoader import nn_imgLoader
    from nn_interactiveTraining import interactiveTraining
    from nn_config import load_config

# Basics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow
import sklearn
import shutil
import json
import time
import cv2
import os

sns.set_style('darkgrid')

# Verzeichnis dieses Moduls (.../src/training) – unabhängig vom Arbeitsverzeichnis.
_TRAIN_ROOT = os.path.dirname(os.path.abspath(__file__))

"""
Das ist die Trainingsklasse, riesig, kümmert sich um das interaktive Trainieren des Modells.

Attributes:
    inited: Description of the attribute.
    train_path: Description of the attribute.
    test_path: Description of the attribute.
    valid_path: Description of the attribute.
    imgLoader: Description of the attribute.
    train_df: Description of the attribute.
    model: Description of the attribute.

Methods:

"""
class nn_Trainer:
    def __init__(self, config=None):
        self.inited = True
        self.config = config if config is not None else load_config()
        self.train_path = r'train'
        self.test_path = r'test'
        self.valid_path = r'valid'
        self.aug_path = r'aug'
        self.imgLoader = nn_imgLoader()
        self.imageSize = tuple(self.config.image_size)
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.ask_epoch = self.config.ask_epoch
        self.learning_rate = self.config.learning_rate
        self.seed = self.config.seed
        self.train_df = None
        self.model = None
        self.classes = None
        self.class_count = None

    def loadLearnData(self):
        for path in [self.train_path, self.test_path, self.valid_path]:
            filepaths = []
            labels = []
            classlist = self.imgLoader.extractClassList(os.path.join(_TRAIN_ROOT, path))
            for klass in classlist:
                classpath = os.path.join(_TRAIN_ROOT, path, klass)
                flist = os.listdir(classpath)
                for f in flist:
                    fpath = os.path.join(classpath, f)
                    filepaths.append(fpath)
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

        print('self.train_df lenght: ', len(self.train_df), '  test_df length: ',
            len(self.test_df), '  self.valid_df length: ', len(self.valid_df))
        # Anzahl der Klassen holen
        self.classes = sorted(list(self.train_df['labels'].unique()))
        self.class_count = len(self.classes)
        print('The number of classes in the dataset is: ', self.class_count)

        groups = self.train_df.groupby('labels')
        print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))

        countlist = []
        classlist = []

        for label in sorted(list(self.train_df['labels'].unique())):
            group = groups.get_group(label)
            countlist.append(len(group))
            classlist.append(label)
            print('{0:^30s} {1:^13s}'.format(label, str(len(group))))

        # Max und Min an Trainingsdatensätzen ermitteln
        max_value = np.max(countlist)
        max_index = countlist.index(max_value)
        max_class = classlist[max_index]
        min_value = np.min(countlist)
        min_index = countlist.index(min_value)
        min_class = classlist[min_index]
        print(max_class, ' Hat die meisten Bilder = ', max_value, ' ', min_class, ' Hat die wenigsten Bilder = ', min_value)

        # Durchschnittshöhe und Breite der Bilder
        ht = 0
        wt = 0
        sample_size = min(100, len(self.train_df))
        # 100 Zufällige Bilder wählen
        train_df_sample = self.train_df.sample(n=sample_size, random_state=123, axis=0)
        for i in range(len(train_df_sample)):
            fpath = train_df_sample['filepaths'].iloc[i]
            img = plt.imread(fpath)
            shape = img.shape
            ht += shape[0]
            wt += shape[1]

        print('Durchsch. Höhe = ', ht // sample_size, ' Durchsch. Breite = ',
              wt // sample_size, 'Verhältnis = ', ht / wt)

    def balance(self, df, n, working_dir):
        df=df.copy()
        print('Anfangslänge des Dataframes : ', len(df))
        aug_dir=os.path.join(_TRAIN_ROOT, working_dir)# Verzeichniss für bearbeitete Bilder

        if os.path.isdir(aug_dir):
            shutil.rmtree(aug_dir)
        os.mkdir(aug_dir)

        for label in df['labels'].unique():
            dir_path=os.path.join(aug_dir,label)
            os.mkdir(dir_path) # Klassenordner erstellen

        # Augmentierung on-the-fly über Keras-Preprocessing-Layer (tf.data-Stil)
        # statt des veralteten ImageDataGenerator.
        augmenter = tensorflow.keras.Sequential([
            tensorflow.keras.layers.RandomFlip('horizontal'),
            tensorflow.keras.layers.RandomRotation(0.06),
            tensorflow.keras.layers.RandomTranslation(0.2, 0.2),
            tensorflow.keras.layers.RandomZoom(0.2),
        ])

        total=0
        groups=df.groupby('labels') # Nach Klassen sortieren

        for label in df['labels'].unique():
            group=groups.get_group(label)
            sample_count=len(group)   # Datensätze ermitteln

            if sample_count< n:
                delta=n - sample_count  # Anzahl der bearbeiteten Bilder
                target_dir=os.path.join(aug_dir, label)
                src_paths=group['filepaths'].tolist()
                msg='{0:40s} für die klasse {1:^30s} erstellt {2:^5s} bearbeitete Bilder'.format(' ', label, str(delta))
                print(msg, '\r', end='') # prints over on the same line

                for i in range(delta):
                    src=src_paths[i % len(src_paths)]
                    raw=tensorflow.io.read_file(src)
                    img=tensorflow.image.decode_image(raw, channels=3, expand_animations=False)
                    img=tensorflow.image.resize(img, self.imageSize)
                    aug=augmenter(tensorflow.expand_dims(img, 0), training=True)[0]
                    aug=tensorflow.cast(tensorflow.clip_by_value(aug, 0, 255), tensorflow.uint8)
                    enc=tensorflow.io.encode_jpeg(aug)
                    tensorflow.io.write_file(os.path.join(target_dir, f'aug-{i}.jpg'), enc)
                    total += 1
        print('Gesamtzahl bearbeiteter Bilder = ', total)

        # create aug_df and merge with train_df to create composite training set ndf
        aug_fpaths=[]
        aug_labels=[]
        classlist=os.listdir(aug_dir)

        for klass in classlist:
            classpath=os.path.join(aug_dir, klass)
            flist=os.listdir(classpath)

            for f in flist:
                fpath=os.path.join(classpath,f)
                aug_fpaths.append(fpath)
                aug_labels.append(klass)

        Fseries=pd.Series(aug_fpaths, name='filepaths')
        Lseries=pd.Series(aug_labels, name='labels')
        aug_df=pd.concat([Fseries, Lseries], axis=1)
        df=pd.concat([df,aug_df], axis=0).reset_index(drop=True)
        print('Die Länge des Dataframes ist nun :', len(df))
        print(df.columns.tolist())
        return df 

    def configureModell(self):
        # Migration von ImageDataGenerator -> keras.utils.image_dataset_from_directory / tf.data.
        AUTOTUNE = tensorflow.data.AUTOTUNE
        train_dir = os.path.join(_TRAIN_ROOT, self.train_path)
        valid_dir = os.path.join(_TRAIN_ROOT, self.valid_path)
        test_dir = os.path.join(_TRAIN_ROOT, self.test_path)
        aug_dir = os.path.join(_TRAIN_ROOT, self.aug_path)

        # Klassennamen (alphabetisch) einmal festlegen und für ALLE Splits erzwingen,
        # damit die Label-Reihenfolge über Train/Valid/Test/Aug konsistent ist.
        self.classes = sorted(
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )
        self.class_count = len(self.classes)

        def _load(directory, shuffle):
            return tensorflow.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='categorical',
                class_names=self.classes,
                color_mode='rgb',
                batch_size=self.batch_size,
                image_size=self.imageSize,
                shuffle=shuffle,
                seed=self.seed,
            )

        train_ds = _load(train_dir, shuffle=True)

        # Balancing-Bilder (falls vorhanden) als zusätzliche Trainingsdaten anhängen.
        if os.path.isdir(aug_dir) and any(
            os.scandir(os.path.join(aug_dir, c)) and os.listdir(os.path.join(aug_dir, c))
            for c in os.listdir(aug_dir)
            if os.path.isdir(os.path.join(aug_dir, c))
        ):
            aug_ds = _load(aug_dir, shuffle=True)
            train_ds = train_ds.concatenate(aug_ds)

        valid_ds = _load(valid_dir, shuffle=False)
        test_ds = _load(test_dir, shuffle=False)

        # Augmentierung on-the-fly nur auf den Trainingsdaten.
        augmenter = tensorflow.keras.Sequential([
            tensorflow.keras.layers.RandomFlip('horizontal'),
            tensorflow.keras.layers.RandomRotation(0.06),
            tensorflow.keras.layers.RandomTranslation(0.2, 0.2),
            tensorflow.keras.layers.RandomZoom(0.2),
        ], name='augmentation')

        train_ds = train_ds.map(
            lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)

        self.train_gen = train_ds.prefetch(AUTOTUNE)
        self.valid_gen = valid_ds.prefetch(AUTOTUNE)
        # Ungemischtes Test-Set separat halten, um y_true zuverlässig zu lesen.
        self.test_gen = test_ds.prefetch(AUTOTUNE)
        self._test_ds_raw = test_ds

        print('number of classes : ', self.class_count, ' classes: ', self.classes)

    def modellMixer(self):
        img_shape=(self.imageSize[0], self.imageSize[1], 3)
        model_name='EfficientNetB3'
        base_model=tensorflow.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max') 

        base_model.trainable=True
        x = base_model.output
        x = tensorflow.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
        x = tensorflow.keras.layers.Dense(256, kernel_regularizer = tensorflow.keras.regularizers.l2(0.016),activity_regularizer=tensorflow.keras.regularizers.l1(0.006),
                        bias_regularizer=tensorflow.keras.regularizers.l1(0.006), activation='relu')(x)
        x = tensorflow.keras.layers.Dropout(rate=.4, seed=123)(x)       
        output=tensorflow.keras.layers.Dense(self.class_count, activation='softmax')(x)
        self.model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=output)
        lr=self.learning_rate # start with this learning rate
        self.model.compile(tensorflow.keras.optimizers.Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 

    def startTraining(self):
        self.ask = interactiveTraining(self.epochs,  self.ask_epoch)
        #self.rlronp=tensorflow.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,verbose=1)
        #self.callbacks=[rlronp, ask]
        self.callbacks=[self.ask]
        return self.model.fit(x=self.train_gen,  epochs=self.epochs, verbose=1, callbacks=self.callbacks,  validation_data=self.valid_gen,
                    validation_steps=None,  shuffle=False,  initial_epoch=0)

    def analyzeTraining(self, tr_data, start_epoch):
        #Plot the training and validation data
        tacc=tr_data.history['accuracy']
        tloss=tr_data.history['loss']
        vacc=tr_data.history['val_accuracy']
        vloss=tr_data.history['val_loss']
        Epoch_count=len(tacc)+ start_epoch
        Epochs=[]
        for i in range (start_epoch ,Epoch_count):
            Epochs.append(i+1)   
        index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
        val_lowest=vloss[index_loss]
        index_acc=np.argmax(vacc)
        acc_highest=vacc[index_acc]
        plt.style.use('fivethirtyeight')
        sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
        vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
        fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))
        axes[0].plot(Epochs,tloss, 'r', label='Training loss')
        axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
        axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
        axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
        axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        
        

    def simplePredict(self):
        # y_true direkt aus dem ungemischten Test-Dataset lesen (tf.data).
        y_true = np.concatenate(
            [y.numpy().argmax(axis=1) for _, y in self._test_ds_raw], axis=0)
        errors=0
        preds=self.model.predict(self.test_gen, verbose=1)
        y_pred=preds.argmax(axis=1)
        tests=len(y_true)
        errors=int(np.sum(y_pred != y_true))

        acc=( 1-errors/tests) * 100
        print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}')
        ypred=np.array(y_pred)
        ytrue=np.array(y_true)
    
        if self.class_count <=30:
            cm = sklearn.metrics.confusion_matrix(ytrue, ypred )
            # plot the confusion matrix
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)       
            plt.xticks(np.arange(self.class_count)+.5, self.classes, rotation=90)
            plt.yticks(np.arange(self.class_count)+.5, self.classes, rotation=0)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.show()
        clr = sklearn.metrics.classification_report(y_true, y_pred, target_names=self.classes, digits= 4) # create classification report
        print("Classification Report:\n----------------------\n", clr)
        return errors, tests
        
    def storeModell(self, errors, tests):
        subject='nuts' 
        acc=str(( 1-errors/tests) * 100)
        index=acc.rfind('.')
        acc=acc[:index + 3]
        save_id= subject + '_' + str(acc) + '.h5' 
        model_dir=os.path.join(_TRAIN_ROOT, "model")
        os.makedirs(model_dir, exist_ok=True)
        model_save_loc=os.path.join(model_dir, save_id)
        print(model_save_loc)
        self.model.save(model_save_loc)
        print ('model was saved as ' , model_save_loc )

        # Klassennamen zusammen mit dem Modell ablegen, damit die Runtime die
        # Label-Reihenfolge nicht aus den Ordnern rekonstruieren muss.
        if self.classes:
            labels_loc=os.path.join(model_dir, subject + '_' + str(acc) + '.labels.json')
            with open(labels_loc, 'w', encoding='utf-8') as fh:
                json.dump(self.classes, fh, ensure_ascii=False, indent=2)
            print('labels were saved as ', labels_loc)
        return model_save_loc

    def exportTFLite(self, errors, tests, quantize=True):
        """Exportiert das trainierte Modell als TensorFlow-Lite-Datei (Edge-Geräte).

        Args:
            errors, tests: für die Namensgebung (Accuracy im Dateinamen).
            quantize: bei True Standard-Optimierung (Dynamic-Range-Quantisierung).

        Returns:
            Pfad zur erzeugten ``*.tflite``-Datei.
        """
        if self.model is None:
            raise RuntimeError("Kein Modell vorhanden – erst trainieren/laden.")

        subject='nuts'
        acc=str(( 1-errors/tests) * 100)
        acc=acc[:acc.rfind('.') + 3]
        model_dir=os.path.join(_TRAIN_ROOT, "model")
        os.makedirs(model_dir, exist_ok=True)

        converter = tensorflow.lite.TFLiteConverter.from_keras_model(self.model)
        if quantize:
            converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_loc=os.path.join(model_dir, f'{subject}_{acc}.tflite')
        with open(tflite_loc, 'wb') as fh:
            fh.write(tflite_model)
        print('TFLite-Modell gespeichert als ', tflite_loc)

        # Labels auch als einfache Textdatei für minimale Edge-Runtimes.
        if self.classes:
            labels_txt=os.path.join(model_dir, f'{subject}_{acc}.labels.txt')
            with open(labels_txt, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(self.classes))
            print('TFLite-Labels gespeichert als ', labels_txt)
        return tflite_loc
   

if __name__ == "__main__":
    print(f"================================= 0. nn_Trainer: Es wurde gestartet & wird initiert ... ======================= ")
    trainer = nn_Trainer()
    print(f"================================= 0. nn_Trainer: Es wurde initiert! =========================================== ")
    print(f"================================= 1. nn_Trainer: Lade Lerndaten ... =========================================== ")
    trainer.loadLearnData()
    print(f"================================= 1. nn_Trainer: Lerndaten geladen! =========================================== ")
    print(f"================================= 2. nn_Trainer: Lerndaten Ausbalancieren ... ================================= ") 
    trainer.balance(trainer.train_df, trainer.config.balance_count, trainer.aug_path)  
    print(f"================================= 2. nn_Trainer: Lerndaten Balanciert! ======================================== ") 
    print(f"================================= 3. nn_Trainer: Setze die Modellparameter ... ================================ ") 
    trainer.configureModell() 
    print(f"================================= 3. nn_Trainer: Modellparameter gesetzt! ===================================== ") 
    print(f"================================= 4. nn_Trainer: Mixe ein vorhandenes Basismodell ein ... ===================== ") 
    trainer.modellMixer()
    print(f"================================= 4. nn_Trainer:  Basismodell eingemixxt! ===================================== ") 
    print(f"================================= 5. nn_Trainer: Starte das Training ... ====================================== ") 
    history = trainer.startTraining()
    print(f"================================= 5. nn_Trainer: Training gestartet -> Bitte Status beobachten! =============== ")
    print(f"================================= 6. nn_Trainer: Führe eine Auswertung des Training durch ... ================= ")
    trainer.analyzeTraining(history,0)
    print(f"================================= 6. nn_Trainer: Trainingshistorie ermittelt! ================================= ")
    print(f"================================= 7. nn_Trainer: Starte eine Vorhersage ... =================================== ")
    errors, tests = trainer.simplePredict()
    print(f"================================= 7. nn_Trainer: Vorhersagen getroffen! ======================================= ")
    print(f"================================= 7. nn_Trainer: Speichere das Modell ... ===================================== ")
    trainer.storeModell(errors, tests)
    print(f"================================= 7. nn_Trainer: Modell gespeichert! ========================================== ")
    print(f"================================= 8. nn_Trainer: Exportiere TensorFlow Lite ... =============================== ")
    trainer.exportTFLite(errors, tests)
    print(f"================================= 8. nn_Trainer: TFLite exportiert! =========================================== ")
    print(f"================================= 8. nn_Trainer: ALLES FERTIG ================================================= ")
