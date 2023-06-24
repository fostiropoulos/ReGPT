from pathlib import Path
from tempfile import TemporaryDirectory
from regpt.compiler import Compiler

from torch.utils.data import DataLoader
from reanalogy.dataset import ReAnalogy
from regpt.model import ReConfig, ReGPT, ReGPTConfig, ReWrapper
from reanalogy import data_path
from unittest.mock import patch
from regpt.utils import get_best_chkpt


def make_artifacts() -> tuple[ReAnalogy, ReGPT]:

    ds = ReAnalogy()
    cfg = ReGPTConfig()
    cfg.vocab_size = ds.vocab_size
    model = ReGPT(cfg)
    return ds, model


def test_model():
    ds, model = make_artifacts()
    dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
    )
    batch = next(iter(dl))

    _, out = model(**batch)
    out.backward()
    assert out.requires_grad and list(model.parameters())[0].grad is not None
    model.zero_grad()


def test_train(tmp_path: str | Path | None = None):
    with patch("reanalogy.ReAnalogy.__len__", return_value=100):
        wrapper = ReWrapper()
        run_config = ReConfig()
        run_config.device = "cpu"
        run_config.amp = False
        run_config.train_config.dataset_path = data_path
        run_config.model_config.n_head = 2
        run_config.model_config.n_layers = 1
        run_config.model_config.embed_dim = 6

        if tmp_path is not None:
            run_config.experiment_dir = Path(tmp_path)
        wrapper.train(run_config=run_config)
        if tmp_path is not None:
            best_chkpt = get_best_chkpt(tmp_path)
            c = Compiler(data_path, best_chkpt)
            c.evaluate(n_samples=1)


if __name__ == "__main__":

    with TemporaryDirectory() as t:
        test_train(t)
