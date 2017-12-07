
""" config system
"""
import numpy as np
import os.path
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.USE_GPU_NMS = True
__C.GPU_ID = 2

def get_output_dir(imdb, net):
    """ Return the directory where experimental artifacts are placed.
        A canonical path is built using the name from an imdb and a network
        (if not None).
    """
    path = os.path.abspath(os.path.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is None:
        return path
    else:
        return os.path.join(path, net.name)


def _merge_two_config(user_cfg, default_cfg):
    """ Merge user's config into default config dictionary, clobbering the
        options in b whenever they are also specified in a.
        Need to ensure the type of two val under same key are the same
        Do recursive merge when encounter hierarchical dictionary
    """
    if type(user_cfg) is not edict:
        return
    for key, val in user_cfg.iteritems():
        # Since user_cfg is a sub-file of default_cfg
        if not default_cfg.has_key(key):
            raise KeyError('{} is not a valid config key'.format(key))

        if type(default_cfg[key]) is not type(val):
            if isinstance(default_cfg[key], np.ndarray):
                val = np.array(val, dtype=default_cfg[key].dtype)
            else:
                raise ValueError(
                     'Type mismatch ({} vs. {}) '
                     'for config key: {}'.format(type(default_cfg[key]),
                                                 type(val), key))
        # Recursive merge config
        if type(val) is edict:
            try:
                _merge_two_config(user_cfg[key], default_cfg[key])
            except:
                print 'Error under config key: {}'.format(key)
                raise
        else:
            default_cfg[key] = val


def cfg_from_file(file_name):
    """ Load a config file and merge it into the default options.
    """
    import yaml
    with open(file_name, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_two_config(yaml_cfg, __C)
