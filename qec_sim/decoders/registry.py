# qec_sim/decoders/registry.py
from qec_sim.core.registry import Registry
from qec_sim.decoders.base import BaseDecoder

decoder_registry: Registry[BaseDecoder] = Registry("decoder")

register_decoder = decoder_registry.register
get_decoder_class = decoder_registry.get

# 하위 호환 유지 (build_decoder 헬퍼)
def build_decoder(name: str, **kwargs) -> BaseDecoder:
    return decoder_registry.get(name)(**kwargs)
