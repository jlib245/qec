# qec_sim/data/registry.py
from qec_sim.core.registry import Registry
from qec_sim.core.interfaces import BasePreprocessor

preprocessor_registry: Registry[BasePreprocessor] = Registry("preprocessor")

register_preprocessor = preprocessor_registry.register
get_preprocessor_class = preprocessor_registry.get
