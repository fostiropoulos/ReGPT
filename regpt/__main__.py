from regpt.model import ReConfig, ReWrapper
import argparse
import torch
from pathlib import Path

if __name__ == "__main__":
    args = argparse.ArgumentParser(
        prog="ReGPT",
        description="Train ReGPT on ReAnalogy",
    )
    args.add_argument(
        "--dataset",
        choices=["reanalogy", "deep", "kb13"],
        required=False,
        default="reanalogy",
    )
    args.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
    )
    args.add_argument(
        "--save_path",
        type=Path,
        required=True,
    )
    args.add_argument(
        "--device",
        choices=[f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"],
        required=False,
        default="cuda",
    )
    args.add_argument("--resume", action="store_true")
    args.add_argument("--regpt", action="store_true")
    args.add_argument("--n_examples", choices=[2, 5, 8, 12], type=int, default=12)
    args.add_argument("--n_layers", choices=[2, 6, 24], type=int, default=6)
    args.add_argument("--batch_size", type=int, default=16)
    kwargs = vars(args.parse_args())
    run_config = ReConfig(
        train_config={
            "dataset": kwargs["dataset"],
            "dataset_path": kwargs["dataset_path"],
            "n_examples": kwargs["n_examples"],
            "epochs": 200,
            "batch_size": 8,
        },
        model_config={
            "n_layers": kwargs["n_layers"],
            "n_head": 12,
            "embed_dim": 16,
        },
        device=kwargs["device"],
        experiment_dir=kwargs["save_path"],
        eval_epoch=1,
    )
    trainer = ReWrapper()
    if run_config.device == "cpu":
        run_config.amp = False
    trainer.train(run_config=run_config, resume=kwargs["resume"])
