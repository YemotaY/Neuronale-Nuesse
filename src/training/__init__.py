# -*- coding: utf-8 -*-
#
#  __init__.py  (training)
#
#  MIT Licensed, 2026. All rights reserved.
#   b59190308@gmail.com
#
"""Trainings-Paket für Neuronale Nüsse 2.0.

Bündelt die Trainingspipeline, den Bild-/Metadaten-Loader, den interaktiven
Trainings-Callback und die Tkinter-GUI zum EXIF-Tagging.
"""

from .nn_imgLoader import nn_imgLoader
from .nn_interactiveTraining import interactiveTraining
from .nn_Trainer import nn_Trainer

__all__ = ["nn_Trainer", "nn_imgLoader", "interactiveTraining"]

# PyTorch-Variante nur exportieren, wenn torch installiert ist.
try:
    from .nn_Trainer_torch import nn_Trainer_torch
    __all__.append("nn_Trainer_torch")
except ImportError:
    pass
