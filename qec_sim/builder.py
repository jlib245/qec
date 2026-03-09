# qec_sim/builder.py
import torch
from qec_sim.data.registry import build_preprocessor
from qec_sim.models.registry import build_model
from qec_sim.decoders.registry import build_decoder

class ComponentBuilder:
    """YAML 설정을 기반으로 전처리기, 모델, 디코더를 조립해주는 전담 팩토리 클래스"""
    
    @staticmethod
    def build_neural_components(config, circuit, num_detectors, device):
        # 1. 전처리기(Preprocessor) 빌드
        prep_config = config.model.preprocessor
        preprocessor = None
        
        if prep_config and 'name' in prep_config:
            prep_name = prep_config.get('name')
            prep_kwargs = prep_config.get('kwargs', {}).copy()
            
            # 런타임 메타데이터 주입
            if "flat" in prep_name:
                prep_kwargs['num_detectors'] = num_detectors
            elif "grid" in prep_name:
                prep_kwargs['detector_coords'] = circuit.get_detector_coordinates()
                
            preprocessor = build_preprocessor(prep_name, **prep_kwargs)

        # 2. 모델(Model) 빌드
        model_kwargs = config.model.kwargs.copy()
        
        if preprocessor:
            if hasattr(preprocessor, 'out_channels'):
                model_kwargs['in_channels'] = preprocessor.out_channels
                model_kwargs['grid_h'] = getattr(preprocessor, 'grid_h', None)
                model_kwargs['grid_w'] = getattr(preprocessor, 'grid_w', None)
            if hasattr(preprocessor, 'output_dim'):
                model_kwargs['input_dim'] = preprocessor.output_dim
                
        model_kwargs['num_observables'] = circuit.num_observables
        model = build_model(config.model.name, **model_kwargs).to(device)

        return preprocessor, model

    @staticmethod
    def build_evaluation_decoder(config, circuit, neural_model=None, preprocessor=None):
        # 3. 디코더(Decoder) 빌드
        decoder_kwargs = config.decoder.model_kwargs.copy()
        decoder_kwargs['error_model'] = circuit.detector_error_model(decompose_errors=True)
        
        if neural_model is not None and preprocessor is not None:
            decoder_kwargs['model'] = neural_model
            decoder_kwargs['preprocessor'] = preprocessor
            
        return build_decoder(config.decoder.name, **decoder_kwargs)