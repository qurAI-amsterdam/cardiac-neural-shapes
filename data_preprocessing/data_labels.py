# Imports
from enum import Enum
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAPPING BETWEEN CARDIAC STRUCTURES AND THEIR LABELS 
# FOR DIFERENT DATASETS AND/OR SEGMENTATION MODELS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# NOTE:
# CTALabelTS['LVBP'] <=> CTALabelTS.LVBP
# CTALabelTS['LVBP'].name --> 'LVBP'
# CTALabelTS['LVBP'].value --> 3
# CTALAbelTS['LVBP'].__class__ --> <enum 'CTALabelTS'>

class CTALabelTS(Enum):
    """
    Suitable for:
     - CTA segmentations used in this project
    """
    BG = 0      # background
    LV = 1      # myocardium
    LABP = 2    # left atrium
    LVBP = 3    # left ventricle
    RABP = 4    # right atrium
    RVBP = 5    # right ventricle
    AOR = 6     # aorta
    PUL = 7     # pulmonary artery    
# --------------------


class MRILabel(Enum):
    """ 
    Suitable for:
     - CMRI segmentations used in this project
    """
    BG = 0
    LVBP = 1
    LV = 2
    RVBP = 3
# --------------------

    
    