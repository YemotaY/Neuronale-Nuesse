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
except ImportError:  # Ausführung als eigenständiges Skript
    from nn_imgLoader import nn_imgLoader
    from nn_interactiveTraining import interactiveTraining

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
    def __init__(self):
        self.inited = True
        self.train_path = r'train'
        self.test_path = r'test'
        self.valid_path = r'valid'
        self.imgLoader = nn_imgLoader()
        self.imageSize = (200,200)
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

        # Besseren Input generieren 
        total=0
        gen = tensorflow.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.2,
                                    height_shift_range=.2, zoom_range=.2)
        groups=df.groupby('labels') # Nach Klassen sortieren

        for label in df['labels'].unique():               
            group=groups.get_group(label)  
            sample_count=len(group)   # Datensätze ermitteln 

            if sample_count< n: 
                aug_img_count=0
                delta=n - sample_count  # Anzahl der bearbeiteten Bilder
                target_dir=os.path.join(aug_dir, label) 
                msg='{0:40s} für die klasse {1:^30s} erstellt {2:^5s} bearbeitete Bilder'.format(' ', label, str(delta))
                print(msg, '\r', end='') # prints over on the same line
                aug_gen=gen.flow_from_dataframe( group,  x_col='filepaths', y_col=None, target_size=self.imageSize,
                                                class_mode=None, batch_size=1, shuffle=False, 
                                                save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                                save_format='jpg')

                while aug_img_count<delta:
                    images=next(aug_gen)            
                    aug_img_count += len(images)
                total +=aug_img_count
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
        batch_size=20 # EfficientNetB3 mit imageSize (200, 200); diese Gr\u00f6\u00dfe sollte keinen Ressourcenfehler ausl\u00f6sen
        trgen = tensorflow.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,rotation_range=20, width_shift_range=.2,
                                        height_shift_range=.2, zoom_range=.2 )
        
        t_and_v_gen = tensorflow.keras.preprocessing.image.ImageDataGenerator()
        msg='{0:70s} for train generator'.format(' ')
        print(msg, '\r', end='') # prints over on the same line
        self.train_gen=trgen.flow_from_dataframe(self.train_df, x_col='filepaths', y_col='labels', target_size=self.imageSize,
                                        class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)

        msg='{0:70s} for valid generator'.format(' ')
        print(msg, '\r', end='') # prints over on the same line
        self.valid_gen=t_and_v_gen.flow_from_dataframe(self.valid_df, x_col='filepaths', y_col='labels', target_size=self.imageSize,
                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

        # for the test_gen we want to calculate the batch size and test steps such that batch_size X test_steps= number of samples in test set
        # this insures that we go through all the sample in the test set exactly once.
        length=len(self.test_df)
        test_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and length/n<=80],reverse=True)[0]  
        self.test_steps=int(length/test_batch_size)
        msg='{0:70s} for test generator'.format(' ')
        print(msg, '\r', end='') # prints over on the same line
        self.test_gen=t_and_v_gen.flow_from_dataframe(self.test_df, x_col='filepaths', y_col='labels', target_size=self.imageSize,
                                        class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)

        # from the generator we can get information we will need later
        self.classes=list(self.train_gen.class_indices.keys())
        class_indices=list(self.train_gen.class_indices.values())
        self.class_count=len(self.classes)
        labels=self.test_gen.labels
        print ( 'test batch size: ' ,test_batch_size, '  test steps: ', self.test_steps, ' number of classes : ', self.class_count)

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
        lr=.001 # start with this learning rate
        self.model.compile(tensorflow.keras.optimizers.Adamax(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy']) 

    def startTraining(self):
        self.epochs=40
        self.ask_epoch=40
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
        y_pred= []
        y_true=self.test_gen.labels
        self.classes=list(self.test_gen.class_indices.keys())
        self.class_count=len(self.classes)
        errors=0
        preds=self.model.predict(self.test_gen, verbose=1)
        tests=len(preds)    
    
        for i, p in enumerate(preds):        
            pred_index=np.argmax(p)         
            true_index=self.test_gen.labels[i]  # labels are integer values        
    
            if pred_index != true_index: # a misclassification has occurred                                           
                errors=errors + 1
                file=self.test_gen.filenames[i]
                print ('file ', file, ' was an error')
            y_pred.append(pred_index)
                
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
   

if __name__ == "__main__":
    print(f"================================= 0. nn_Trainer: Es wurde gestartet & wird initiert ... ======================= ")
    trainer = nn_Trainer()
    print(f"================================= 0. nn_Trainer: Es wurde initiert! =========================================== ")
    print(f"================================= 1. nn_Trainer: Lade Lerndaten ... =========================================== ")
    trainer.loadLearnData()
    print(f"================================= 1. nn_Trainer: Lerndaten geladen! =========================================== ")
    print(f"================================= 2. nn_Trainer: Lerndaten Ausbalancieren ... ================================= ") 
    trainer.balance(trainer.train_df, 200, r'aug')  
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
    print(f"================================= 7. nn_Trainer: ALLES FERTIG ================================================= ")
