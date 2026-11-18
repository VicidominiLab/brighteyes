# -*- coding: utf-8 -*-
# Origin: VicidominiLab/BrightEyes-FFS, src/brighteyes_ffs/tools/savevar.py at f6d640d.
"""
Created on Fri Apr 12 12:01:51 2019

@author: SPAD-fcs
"""

import pickle
from .checkfname import checkfname


class _LegacyUnpickler(pickle.Unpickler):
    """Resolve saved spectroscopy classes in their current package."""

    def find_class(self, module: str, name: str) -> object:
        """Map the former package prefix before resolving a saved class."""
        if module == 'brighteyes_ffs' or module.startswith('brighteyes_ffs.'):
            module = module.replace('brighteyes_ffs', 'brighteyes.spectroscopy', 1)
        return super().find_class(module, name)


def savevar(var, fname):
    fname = checkfname(fname, 'pickle')
    with open(fname, 'wb') as f:
        pickle.dump(var, f)


def openvar(fname):
    fname = checkfname(fname, 'pickle')
    with open(fname, 'rb') as f:
        data = _LegacyUnpickler(f).load()
        return data
