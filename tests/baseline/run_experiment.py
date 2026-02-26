import argparse
from qec_sim.core.pipelines import EvaluationPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_mwpm.yaml")
    args = parser.parse_args()
    
    # 파이프라인 생성 및 실행
    pipeline = EvaluationPipeline(args.config)
    pipeline.run()