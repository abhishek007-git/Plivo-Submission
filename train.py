import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)     
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)  
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        prog = min(1.0, prog)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cos)

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
