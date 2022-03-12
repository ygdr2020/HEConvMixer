from .hconvmixer import HConvMixer


import torch.nn as nn


def build_model():

    model = HConvMixer(embed_dim=128, patch_size=1, kernel_size=[9, 7, 5], n_classes=10, depth=[3, 3 ,2], r=[4, 4, 4])

    

    return model
