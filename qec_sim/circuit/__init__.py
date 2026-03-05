# qec_sim/circuit/__init__.py
from .registry import register_circuit, build_circuit
from . import builder  # <--- 이 줄이 추가되어야 builder.py의 @register가 실행됨