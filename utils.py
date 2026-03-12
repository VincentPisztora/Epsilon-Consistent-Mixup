# -*- coding: utf-8 -*-
"""

"""


import tensorflow as tf
from tensorflow.python.client import device_lib

###############################################################################
###############################################################################
###############################################################################
### Utils
###############################################################################
###############################################################################
###############################################################################

def get_available_gpus():
    global _GPUS
    if _GPUS is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        local_device_protos = device_lib.list_local_devices(session_config=config)
        _GPUS = tuple([x.name for x in local_device_protos if x.device_type == 'GPU'])
    return _GPUS

def get_config():
    config = tf.ConfigProto()
    if len(get_available_gpus()) > 1:
        config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    return config