# -*- coding: utf-8 -*-
# Origin: VicidominiLab/BrightEyes-FFS, src/brighteyes_ffs/tools/goto.py at f6d640d.
"""
Change current working directory to the folder given as input
"""

import os

def goto(directory):
    """Change the working directory to the supplied path."""
    os.chdir(directory)
