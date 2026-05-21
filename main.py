import argparse
import os
from dotenv import load_dotenv

# CUDA_VISIBLE_DEVICES가 torch의 CUDA 컨텍스트 생성 전에 적용되어야 함.
load_dotenv()

from qec_sim.trainer.pipeline import TrainingPipeline
from qec_sim.trainer.eval_pipeline import EvaluationPipeline


def _register_genai_dataset(config, path: str, role: str, seed):
    """gen 직후 sidebar Datasets 탭에 entity 등록.

    train run 안에서 호출하면 mlflow 가 그 run 을 'GenAI dataset 평가' 로 오분류해서
    Evaluation runs 사이드바에 잘못 잡힘 → gen 모드에서 분리 호출.
    name 컨벤션은 MLflowCallback._log_dataset_ref 와 동일.
    """
    import hashlib, numpy as np
    import mlflow
    from mlflow.genai import datasets as gds

    if not os.path.exists(path):
        return

    # sha256 (sidecar cache 재사용)
    sc = path + ".sha256"
    if os.path.exists(sc) and os.path.getmtime(sc) >= os.path.getmtime(path):
        digest = open(sc).read().strip()
    else:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        try:
            with open(sc, 'w') as f:
                f.write(digest)
        except OSError:
            pass

    # n_samples
    try:
        with np.load(path, allow_pickle=False) as d:
            if 'observables' in d.files:
                n_samples = int(len(d['observables']))
            elif 'logical_outcomes' in d.files:
                n_samples = int(len(d['logical_outcomes']))
            else:
                n_samples = None
    except Exception:
        n_samples = None

    code = config.code
    basename = os.path.basename(path).rsplit('.', 1)[0]
    parts = [basename, f"r{code.rounds}"]
    if n_samples is not None:
        parts.append(f"n{n_samples}")
    if seed is not None:
        parts.append(f"s{seed}")
    name = "_".join(parts)

    # experiment resolve (auto-derive 적용)
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    exp_name = config.mlflow.experiment_name or f"{code.name}_d{code.distance}"
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(
            exp_name,
            artifact_location=config.mlflow.artifact_location,
        )
    else:
        exp_id = exp.experiment_id

    tags_dict = {
        "code": code.name,
        "distance": str(code.distance),
        "rounds": str(code.rounds),
        "path": path,
        "sha256": digest,
        "size": str(os.path.getsize(path)),
        "role": role,
    }
    if n_samples is not None:
        tags_dict["n_samples"] = str(n_samples)
    if seed is not None:
        tags_dict["gen_seed"] = str(seed)

    try:
        existing = [d for d in gds.search_datasets(experiment_ids=[exp_id]) if d.name == name]
        if existing:
            gds.set_dataset_tags(existing[0].dataset_id, tags_dict)
            print(f"  [MLflow] sidebar dataset entity 갱신: {name}")
        else:
            ds = gds.create_dataset(name=name, experiment_id=exp_id, tags=tags_dict)
            print(f"  [MLflow] sidebar dataset entity 생성: {name} (id={ds.dataset_id})")
    except Exception as e:
        print(f"  [MLflow] sidebar dataset entity 등록 실패 ({type(e).__name__}: {e})")


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
    parser.add_argument("--seed", type=int, default=None,
                        help="[gen 모드] np.random/python random 의 seed. 미지정 시 config.training.seed 사용. 둘 다 None 이면 unreproducible (워닝)")
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

        # gen seed: CLI override > config.training.seed > None
        gen_seed = args.seed if args.seed is not None else config.training.seed
        if gen_seed is not None:
            import random, numpy as np
            random.seed(gen_seed)
            np.random.seed(gen_seed)
            print(f"Gen seed: {gen_seed}")
        else:
            print("⚠ Gen seed 미지정 → 재현 불가. --seed 또는 config.training.seed 명시 권장.")

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
            seed=gen_seed,
        )
        gen.generate_and_save(
            shots=args.val_shots,
            save_dir=os.path.dirname(val_path) or ".",
            filename=os.path.splitext(os.path.basename(val_path))[0],
            batch_size=chunk_size,
            seed=gen_seed,
        )

        # mlflow 활성 시 sidebar Datasets 탭용 entity 도 같이 등록 (active run 없는 상태에서)
        if config.mlflow.enable:
            _register_genai_dataset(config, train_path, "train", gen_seed)
            _register_genai_dataset(config, val_path,   "val",   gen_seed)


if __name__ == "__main__":
    main()
