# qec_sim/data/preprocessors/__init__.py
#
# Preprocessor registry is populated as a side effect of importing each
# submodule (the @register_preprocessor decorator runs at import time).
# Class names are also re-exported here for backward compatibility with
# `from qec_sim.data.preprocessors import X` style imports.

from . import basic
from . import qct
from . import gap_routed_cnn
from . import gap_routed_gnn

from .basic import (
    SpatialGridPreprocessor,
    SoftGridPreprocessor,
    FlatPreprocessor,
)
from .qct import QCTPreprocessor
from .gap_routed_cnn import (
    GapRoutedCNNPreprocessor,
    GapRoutedCNNFilteredPreprocessor,
    Syndrome3DPreprocessor,
    Syndrome3DFilteredPreprocessor,
    CachedGapRoutedCNNPreprocessor,
    CachedSyndrome3DPreprocessor,
)
from .gap_routed_gnn import (
    GapRoutedGNNPreprocessor,
    GapRoutedGNNFilteredPreprocessor,
    SyndromeGNNPreprocessor,
    SyndromeGNNFilteredPreprocessor,
    CachedGapRoutedGNNPreprocessor,
    CachedSyndromeGNNPreprocessor,
    CachedGapRoutedGNNCNNFeatPreprocessor,
)
