from hmr4d.utils.preproc.tracker import Tracker
from hmr4d.utils.preproc.vitfeat_extractor import Extractor
from hmr4d.utils.preproc.vitpose import VitPoseExtractor
# This adapter reuses the existing camera; optional SimpleVO is not imported.

try:
    from hmr4d.utils.preproc.slam import SLAMModel
except:
    pass
