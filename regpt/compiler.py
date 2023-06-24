import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from reanalogy.dataset import (
    BOS_TOKEN,
    SEP_TOKEN,
    ReAnalogy,
    gen_regex,
)
from regpt.model import ReGPT, load_model


class Compiler:
    def __init__(self, root_path, model_checkpoint) -> None:
        super().__init__()

        self.base_model, self.cfg = self.load_model(model_checkpoint)
        self.dataset: ReAnalogy = ReAnalogy(
            root_path,
            split="val",
            dataset_name=self.cfg.train_config.dataset,
            return_regex=self.cfg.train_config.regpt,
            n_examples=self.cfg.train_config.n_examples,
            check_match=True,
        )

    def load_model(self, checkpoint):
        model, cfg = load_model(checkpoint)
        return model, cfg

    def _generate(self, model: ReGPT, s, top_k: int = 5, n_samples=5):
        with torch.no_grad():
            gen_samples = model.generate(s, top_k=top_k, n_samples=n_samples)
        decoded_samples = [self.decode(_s) for _s in gen_samples]
        return decoded_samples

    @torch.no_grad()
    def reconstruct(self, seq):

        recon_seq = self.base_model._forward(seq).logits.argmax(-1).cpu().numpy()
        recon_seq = np.concatenate([[BOS_TOKEN], recon_seq])
        seperator = torch.argwhere(seq == SEP_TOKEN).max()
        recon_seq[seperator] = SEP_TOKEN
        return self.dataset.decode_regex(recon_seq)

    def decode(self, seq):
        return self.dataset._decode(seq.cpu().numpy())

    @torch.no_grad()
    def generate(
        self,
        batch,
        device: str | None = None,
        held_out_examples: int = 0,
        top_k: int = 5,
    ):
        model = self.base_model
        if device is not None:
            model.to(device)
        model.eval()
        if self.dataset.return_regex:
            *str_examples, ref_regex = self.decode(batch["seq"])
        else:
            str_examples = self.decode(batch["seq"])
            ref_regex = batch["regex"]

        starting_examples = self._make_starting_examples(
            str_examples, held_out_examples=held_out_examples
        )

        if device is not None:
            starting_examples = starting_examples.to(device)
        decoded_samples = self._generate(
            model, starting_examples, top_k=top_k, n_samples=5
        )
        rows = []
        if self.dataset.score(ref_regex, str_examples).mean() != 1:
            print(f"Invalid score for {ref_regex}, {str_examples}.")
            return pd.DataFrame([])
        for *_examples, _regex in decoded_samples:
            # Score to original examples - GT
            row = {
                "ref_regex": ref_regex,
                "top_k": top_k,
            }

            if self.dataset.return_regex:
                _score = self.dataset.score(_regex, str_examples)
                row["gen_regex"] = _regex
                row["gen_regex_score"] = _score
                row["recon_regex"] = self.reconstruct(batch["seq"].to(device))

                row["regex_equiv"] = self.dataset.score(
                    _regex,
                    gen_regex(ref_regex, num_samples=20, max_attempts=100),
                )
            else:
                _examples.append(_regex)

            # GT examples
            for i, example in enumerate(str_examples):
                row[f"ref_ex_{i}"] = example

            for i in range(len(str_examples) - held_out_examples, len(str_examples)):
                if i >= len(_examples):
                    row[f"gen_ex_{i}"] = None
                    row[f"gen_ex_{i}/ref_regex_score"] = None
                    continue
                row[f"gen_ex_{i}"] = _examples[i]
                row[f"gen_ex_{i}/ref_regex_score"] = self.dataset.score(
                    ref_regex, [_examples[i]]
                )

            rows.append(row)
        return pd.DataFrame(rows)

    def _make_starting_examples(self, str_examples, held_out_examples):
        return torch.concat(
            [torch.tensor([BOS_TOKEN]).long()]
            + [
                torch.tensor(
                    self.dataset._concat_examples(
                        [
                            str_examples[i]
                            for i in range(len(str_examples) - held_out_examples)
                        ]
                    )
                ).long()
            ]
        )

    @torch.no_grad()
    def evaluate(self, device: str | None = None, top_k=5, n_samples=10):

        scores = []

        if not self.cfg.train_config.regpt:
            held_out_examples = 1
        else:
            held_out_examples = 0
        eval_idxs = np.random.permutation(len(self.dataset))

        t = tqdm(total=min(n_samples, len(self.dataset)))
        for i, idx in enumerate(eval_idxs):
            batch = self.dataset[idx]
            scores.append(
                self.generate(
                    batch=batch,
                    device=device,
                    held_out_examples=held_out_examples,
                    top_k=top_k,
                )
            )
            t.update(1)
            if i > n_samples:
                break

        return pd.concat(scores)
