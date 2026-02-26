import argparse
from qec_sim.core.pipelines import TrainingPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment_mlp.yaml")
    args = parser.parse_args()
    
    # 파이프라인 생성 및 실행
    pipeline = TrainingPipeline(args.config)
    pipeline.run()