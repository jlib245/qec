# main.py
from qec_sim.trainer.pipeline import TrainingPipeline, EvaluationPipeline

def main():
    config_file = "configs/example_jung.yaml"

    print("1. 딥러닝 모델 학습 파이프라인 시작")
    train_pipeline = TrainingPipeline(config_file)
    train_pipeline.run()

    # 학습이 끝나면 weight_path를 방금 저장된 best_model.pth로 업데이트했다고 가정
    print("\n2. 디코더 평가 파이프라인 시작")
    eval_pipeline = EvaluationPipeline(config_file)
    eval_pipeline.config.decoder.weight_path = train_pipeline.workspace["best_model"]

    eval_pipeline.run()

if __name__ == "__main__":
    main()