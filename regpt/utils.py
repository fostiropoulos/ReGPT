from pathlib import Path


def get_best_chkpt(tmp_path, model_name="ReGPT_*"):
    chkpts = list(Path(tmp_path).rglob(f"**/{model_name}.pt"))
    best_chkpt = sorted(chkpts)[-1]
    return best_chkpt
