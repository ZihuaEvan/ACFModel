from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .AttModel import *
from .TransformerModel import TransformerModel
from .ACFModel import ACFModel


def setup(opt):

    # Transformer
    if opt.caption_model == 'transformer':
        model = TransformerModel(opt)
    elif opt.caption_model == 'ACF':
        model = ACFModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
