from dataclasses import dataclass, fields

import yaml

@dataclass
class TrainConfig: 
    batch_size: int = 32
    grad_accum_steps: int = 4
    max_steps: int = 1000
    warmup_steps: int = 100
    lr_max: float = 3e-4
    min_lr_ration: float = 0.1

    weight_decay = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_interval: int = 20
    eval_interval: int = 100
    eval_iters: int = 10
    checkpoint_interval: int = 100

    seed: int = 1337
    precision: str = "auto"
    out_dir: str = "checkpoints"

    def tokens_per_step(self, block_size):
        return self.batch_size * self.grad_accum_steps * block_size

    def load_train_config(path):

        with open(path, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)
        known = {f.name for f in fields(TrainConfig)}

        return TrainConfig(**{k: v for k, v in values.items() if k in known})
    