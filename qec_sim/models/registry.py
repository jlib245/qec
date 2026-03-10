# qec_sim/models/registry.py
from qec_sim.core.registry import Registry
from qec_sim.core.interfaces import BaseQECModel

model_registry: Registry[BaseQECModel] = Registry("model")

register_model = model_registry.register
get_model_class = model_registry.get
