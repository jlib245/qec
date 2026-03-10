import argparse
from qec_sim.trainer.pipeline import TrainingPipeline
from qec_sim.trainer.eval_pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description="QEC Simulation Framework")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="설정 파일(YAML) 경로")
    parser.add_argument("-m", "--mode", type=str,
                        choices=["train", "eval"],
                        help="실행 모드: train(기본) 또는 eval")
    parser.add_argument("--model", type=str, default=None,
                        help="[eval 모드] 평가할 모델 가중치 경로 (.pth)")
    parser.add_argument("--shots", type=int, default=10000,
                        help="[eval 모드] 노이즈 설정당 평가 샷 수 (기본: 10000)")
    args = parser.parse_args()

    if args.mode == "train":
        TrainingPipeline(config_path=args.config).run()

    elif args.mode == "eval":
        if args.model is None:
            parser.error("--mode eval 에는 --model 경로가 필요합니다.")
        EvaluationPipeline(
            config_path=args.config,
            model_path=args.model,
        ).run(shots=args.shots)


if __name__ == "__main__":
    main()
