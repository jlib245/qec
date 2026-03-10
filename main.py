import argparse
from qec_sim.trainer.pipeline import TrainingPipeline

def main():
    parser = argparse.ArgumentParser(description="QEC Simulation Training Pipeline")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, 
        help="학습 설정 파일(YAML)의 경로를 입력하세요."
    )
    args = parser.parse_args()

    # 우리가 만든 파이프라인 객체 생성 및 실행
    pipeline = TrainingPipeline(config_path=args.config)
    pipeline.run()

if __name__ == "__main__":
    main()