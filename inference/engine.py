"""
Kechikish (latency) masalasida eng muhim qoida shuki, 
model va tokenizator har bir so'rovda emas, 
balki faqat bir marta — '__init__' bosqichida yuklanishi kerak. 
10 million parametrli modelni diskdan yuklash sezilarli vaqt (yuzlab millisekundlar) talab qiladi; 
agar bu har bir API chaqiruvida amalga oshirilsa, asosiy vaqt aynan shu jarayonga sarflanib ketadi.

Bu yerda ataylab KV-kesh (KV cache) ishlatilmagan.
Buning o‘rniga, har bir yangi token hosil qilinayotganda butun ketma-ketlik model orqali qayta 
o‘tkaziladi (avvalgi kalit/qiymatlar qayta ishlatilmaydi).
Qisqa so‘rovlardan (prompt) tashqari holatlarda bu yondashuv hisoblash resurslarini
kvadratik darajada behuda sarflashga olib keladi. Hozirgi kichik model uchun — ayniqsa, 
faqat CPU mavjud bo‘lgan bepul xosting muhitida — bu holat hali jiddiy to‘siq (bottleneck) hisoblanmaydi.
Biroq, kattaroq modellar nima uchun KV-keshga muhtoj bo‘lishini aniq tushuntira olish foydalidir.
"""

import time
from pathlib import Path

import torch

from src.checkpoint import load_checkpont
from src.model import OwnLLM
from src.model_config import ModelConfig
from src.tokenizer import Tokenizer
from inference.sampling import sample_next_token

class InferenceEngine:
    def __init__(self, config_path, checkpoint_path, tokenizer_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.config = ModelConfig.load_config(config_path)
        self.model = OwnLLM(self.config).to(self.device)
        load_checkpont(checkpoint_path, self.model, map_location=self.device)
        self.model.eval()

        self.tokenizer = Tokenizer().load(tokenizer_path)
        self.eos_id = self.tokenizer.eos_id()

        print("Inference engine ready: {:,} parameters on {}".format(
            self.model.num_parameters(), self.device
        ))

    @torch.no_grad()
    def generate(self, prompt, max_tokens=50, temperature=0.8, top_k=50,
                 top_p=0.95, repitition_penalty=1.1):
        """Returns (text, tokens_generated, latency_ms)."""
        start = time.perf_counter()

        ids = self.tokenizer.encode(prompt)
        generated = list(ids)
        idx = torch.tensor([ids], device=self.device)

        for _ in range(max_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self.model(idx_cond)
            next_logits = logits[0, -1, :]

            next_id = sample_next_token(
                next_logits, generated,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repitition_penalty=repitition_penalty,
            )
            if next_id == self.eos_id:
                break
            generated.append(next_id)
            idx = torch.cat(
                [idx, torch.tensor([[next_id]], device=self.device)], dim=1
            )

        text = self.tokenizer.decode(generated)
        latency_ms = (time.perf_counter() - start) * 1000
        tokens_generated = len(generated) - len(ids)
        return text, tokens_generated, latency_ms


# Order matters: a machine that has just trained has best.pt (the full
# training checkpoint, gitignored), and that should win because it is the
# freshest thing that machine produced. A fresh `git clone` has only
# model.pt - the inference-only weights we commit - so the fallback is what
# makes "clone the repo and generate" work with no flags at all.
CHECKPOINT_CANDIDATES = (Path("checkpoints/best.pt"), Path("checkpoints/model.pt"))


def resolve_checkpoint(explicit=None):
    """Return a checkpoint path that actually exists on this machine."""
    if explicit is not None:
        return Path(explicit)

    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No checkpoint found. Looked for: {}. Train one with "
        "scripts/train.py, or pass --checkpoint explicitly.".format(
            ", ".join(str(c) for c in CHECKPOINT_CANDIDATES)
        )
    )


def default_engine():
    """Convenience constructor using the repo's standard file layout."""
    return InferenceEngine(
        config_path=Path("configs/run_01.yaml"),
        checkpoint_path=resolve_checkpoint(),
        tokenizer_path=Path("tokenizer/tokenizer.json"),
    )
