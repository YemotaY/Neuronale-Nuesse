# -*- coding: utf-8 -*-
#
#  nn_imgLoader.py
#
#  Created by YemotaY on 2026-07-17.
#  MIT Licensed, 2026. All rights reserved.
#
#   b59190308@gmail.com
#
# pylint: disable-msg=F0401

from PIL import Image
from PIL.ExifTags import TAGS
from os import path, getcwd, listdir, walk

'''
Diese Klasse kümmert sich darum dass die Bilder + Meta geladen werden fuer den Trainer.

Methoden:
loadSingle          -> Lädt ein einzelnes Bild mit einem Pfad + Name
loadArrayBatched    -> Lädt aus einem gegeben Ordner mehrere Bilder nach dem Namen-Array
loadFolder          -> Lädt den ganzen Ordner (Rekursivität bestimmbar)
list_recursive_walk -> Extrahiert rekursiv alle Bilder eines Ordners + Unterordnern

Attribute:
statusCode : int    -> Statuscode der letzten Methoden ausführung (0=GOOD,1=ERROR_X,..)

Abhängigkeiten:
pillow-12.3.0
'''
class nn_imgLoader:
    statusCode : int = 0
    def __init__(self):
        self.inited : bool = True
        self.images : list = []

          
    def loadSingle(self, imgPath):
        try:
            if(imgPath == ""): raise Exception(f"Missing Path for loadSingle:{imgPath}")
            image = Image.open(imgPath)
            obj = {"imgObj":image,"format":image.format,"size":image.size,"mode":image.mode,"tags":image.getexif()}
            image.load()  # Pixeldaten laden, damit der Datei-Handle geschlossen wird
            self.images.append(obj)
            #print(f"Image loaded : {obj}")
        except Exception as e:
            print(f"nn_imgLoader-loadSingle Error: {e}")
            self.statusCode = 1
            return None
        self.statusCode = 0
        return obj

    def loadArrayBatched(self, trgPath, trgArray):
        images = []
        try:
            if(trgPath=="" or len(trgArray)==0): raise Exception(f"loadArrayBatched: Missing Path: {trgPath} or empty targetArray: {trgArray} ")
            for img in trgArray:
                loaded = self.loadSingle(path.join(trgPath , img))
                if loaded is not None:
                    images.append(loaded)
        except Exception as e:
            print(f"Error: {e}")
        print(f"loadArrayBatched | Did load {len(images)} images from {trgPath}")
        return images
    
    def loadFolder(self, trgPath, recursive=False):
        images = []
        try:
            if(trgPath==""): raise Exception(f"loadFolder: Missing Path: {trgPath} ")
            if(not recursive):
                inputs = [path.join(trgPath, i) for i in listdir(trgPath)
                          if path.isfile(path.join(trgPath, i))]
            else:
                inputs = self.list_recursive_walk(trgPath)
            for img in inputs:
                loaded = self.loadSingle(img)
                if loaded is not None:
                    images.append(loaded)
        except Exception as e:
            print(f"Error: {e}")
        print(f"loadFolder | Did load {len(images)} images from {trgPath}, recursive: {recursive}")    
        return images 

    def extractClassList(self,trgPath):
        return listdir(trgPath)

    def list_recursive_walk(self,trgPath):
        f = []
        for root, dirs, files in walk(trgPath):
            for file in files:
                f.append(path.join(root, file))
        return f

if __name__ == "__main__":
    newLoader = nn_imgLoader()

    # Tests / Showcases
    print("\nTestload loadSingle :\n")
    newLoader.loadSingle(path.join("src","training","pepefrog.jpg"))                                # Finished
    print("\nTestload loadArrayBatched :\n")
    newLoader.loadArrayBatched(path.join("src","img","DE","mandeln"),["1.jpg","3.jpg","5.jpg",])    # Finished
    print("\nTestload loadFolder non recursive :\n")
    newLoader.loadFolder(path.join("src","img","DE","mandeln"),False)                               # Finished
    print("\nTestload loadFolder recursive :\n")
    newLoader.loadFolder(path.join("src","img","DE"),True)                               # Finished

