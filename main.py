import argparse
import os
from dotenv import load_dotenv

# CUDA_VISIBLE_DEVICES가 torch의 CUDA 컨텍스트 생성 전에 적용되어야 함.
load_dotenv()

from qec_sim.trainer.pipeline import TrainingPipeline
from qec_sim.trainer.eval_pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(description="QEC Simulation Framework")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="설정 파일(YAML) 경로")
    parser.add_argument("-m", "--mode", type=str,
                        choices=["train", "eval", "gen"],
                        help="실행 모드: train, eval, 또는 gen(오프라인 데이터셋 생성)")
    parser.add_argument("--model", type=str, default=None,
                        help="[eval 모드] 평가할 모델 가중치 경로 (.pth)")
    parser.add_argument("--shots", type=int, default=None,
                        help="[eval 모드] 노이즈 설정당 평가 샷 수 (필수)")
    parser.add_argument("--train-shots", type=int, default=None,
                        help="[gen 모드] 학습용 샷 수 (필수)")
    parser.add_argument("--val-shots", type=int, default=None,
                        help="[gen 모드] 검증용 샷 수 (필수)")
    args = parser.parse_args()

    if args.mode == "train":
        TrainingPipeline(config_path=args.config).run()

    elif args.mode == "eval":
        if args.shots is None:
            raise ValueError("eval 모드는 --shots를 명시해야 합니다.")
        EvaluationPipeline(
            config_path=args.config,
            model_path=args.model,
        ).run(shots=args.shots)

    elif args.mode == "gen":
        if args.train_shots is None or args.val_shots is None:
            raise ValueError("gen 모드는 --train-shots / --val-shots를 명시해야 합니다.")
        from qec_sim.config.schema import ExperimentConfig
        from qec_sim.data.generator import DatasetGenerator

        config = ExperimentConfig.from_yaml(args.config)
        gen = DatasetGenerator.from_config(config)

        train_path = config.training.train_path
        val_path = config.training.val_path
        if train_path is None or val_path is None:
            raise ValueError("gen 모드는 training.train_path / training.val_path 가 yaml에 명시돼야 합니다.")

        chunk_size = config.training.chunk_size 
        gen.generate_and_save(
            shots=args.train_shots,
            save_dir=os.path.dirname(train_path) or ".",
            filename=os.path.splitext(os.path.basename(train_path))[0],
            batch_size=chunk_size,
        )
        gen.generate_and_save(
            shots=args.val_shots,
            save_dir=os.path.dirname(val_path) or ".",
            filename=os.path.splitext(os.path.basename(val_path))[0],
            batch_size=chunk_size,
        )


if __name__ == "__main__":
    main()
