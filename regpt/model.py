from ablator.main.configs import RunConfig
import torch
from ablator import (
    Derived,
    ModelConfig,
    ModelWrapper,
    TrainConfig,
    OptimizerConfig,
    RunConfig,
    Literal,
    ParallelConfig,
)
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

from reanalogy.dataset import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, ReAnalogy


def load_model(checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu")
    cfg = ReConfig(**state_dict["run_config"])
    model = ReGPT(cfg.model_config)
    model.load_state_dict(state_dict["model"])
    model.eval()
    return model, cfg


class ReGPTConfig(ModelConfig):
    vocab_size: Derived[int]
    embed_dim: int = 12
    n_head: int = 12
    n_layers: int = 3
    max_seq_len: Derived[int]


class ReTrainConfig(TrainConfig):
    dataset: Literal["reanalogy", "kb13", "deep"] = "reanalogy"
    dataset_path: Derived[Path]
    regpt: bool = True
    batch_size = 128
    epochs = 1
    optimizer_config = OptimizerConfig(name="adam", arguments={"lr": 0.001})
    n_examples: int = 5


class RePConfig(ParallelConfig):
    train_config: ReTrainConfig = ReTrainConfig()
    model_config: ReGPTConfig = ReGPTConfig()


class ReConfig(RunConfig):
    train_config: ReTrainConfig = ReTrainConfig()
    model_config: ReGPTConfig = ReGPTConfig()
    verbose = "tqdm"


class ReGPT(nn.Module):
    def __init__(self, config: ReGPTConfig) -> None:
        super().__init__()
        # Initializing a BERT bert-base-uncased style configuration
        self.hidden_dim = hidden_dim = config.embed_dim * config.n_head
        self.config = config
        self.max_seq_len = config.max_seq_len
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_embd=hidden_dim,
            n_layer=config.n_layers,
            n_head=config.n_head,
            n_positions=self.max_seq_len
            # add_cross_attention=True,
        )
        self.model = GPT2LMHeadModel(gpt2_config)

    def _forward(self, seq, output_hidden_states=True):
        attention_mask = (seq != PAD_TOKEN).float()
        out = self.model(
            seq,
            labels=seq,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            # encoder_attention_mask=encoder_attention_mask,
        )
        return out

    def forward(self, seq: torch.Tensor, regex: str | None = None):
        out = self._forward(seq, False)
        return None, out.loss

    @torch.no_grad()
    def generate(self, examples, n_samples=5, top_k=5):
        if examples is not None:
            if len(examples.shape) == 1:
                examples = examples[None, :]
        regexes = []
        assert not self.model.training
        for i in range(n_samples):
            out = self.model.generate(
                inputs=examples,
                do_sample=True,
                max_length=self.max_seq_len,
                top_k=top_k,
                eos_token_id=EOS_TOKEN,
                bos_token_id=BOS_TOKEN,
                pad_token_id=PAD_TOKEN,
                suppress_tokens=[BOS_TOKEN, PAD_TOKEN],
            )[0].squeeze()
            padded_regex = torch.nn.functional.pad(
                out, (0, self.max_seq_len - len(out)), "constant", PAD_TOKEN
            )
            regexes.append(padded_regex.cpu())
        return torch.stack(regexes)


class ReWrapper(ModelWrapper):
    def __init__(self):
        super().__init__(ReGPT)

    def make_dataloader_train(self, run_config: ReConfig):
        ds = ReAnalogy(
            run_config.train_config.dataset_path,
            split="train",
            dataset_name=run_config.train_config.dataset,
            n_examples=run_config.train_config.n_examples,
        )
        run_config.model_config.vocab_size = ds.vocab_size

        return DataLoader(
            ds,
            batch_size=self.run_config.train_config.batch_size,
            shuffle=True,
            # num_workers=20,
        )

    def make_dataloader_val(self, run_config: ReConfig):
        ds = ReAnalogy(
            run_config.train_config.dataset_path,
            split="val",
            dataset_name=run_config.train_config.dataset,
            n_examples=run_config.train_config.n_examples,
        )
        run_config.model_config.vocab_size = ds.vocab_size
        return DataLoader(
            ds,
            batch_size=run_config.train_config.batch_size,
            shuffle=True,
            # num_workers=20,
        )

    def config_parser(self, run_config: ReConfig):
        run_config.model_config.max_seq_len = self.train_dataloader.dataset.max_seq_len
        return run_config
