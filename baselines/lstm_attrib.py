#!/usr/bin/env python3
"""Bi-LSTM attribution baseline on BiBE .trace files.

Same protocol as examples/train_real.rs: sorted paths, 80/20 split,
cause-supervised (cross-entropy over positions), Hit@1/3 + MRR on anomalous
test traces, mean +/- std over 5 seeds. Window = first 64 events.
"""
import glob
import os
import re
import statistics
import sys

import torch
import torch.nn as nn

WINDOW, EPOCHS, HIDDEN, SEEDS = 64, 30, 64, [7, 42, 99, 1234, 2025]


def parse_trace(path):
    events, label = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                m = re.match(r"# label=anomalous root_cause=(\d+) cause=(\d+)", line)
                if m:
                    label = (int(m.group(1)), int(m.group(2)))
            elif line:
                parts = line.split()
                events.append((parts[0], int(parts[7])))
    return events[:WINDOW], label


def load(traces_dir):
    paths = sorted(glob.glob(os.path.join(traces_dir, "*.trace")))
    assert paths, f"no .trace files in {traces_dir}"
    data = [parse_trace(p) for p in paths]
    vocab = {"<PAD>": 0}
    for events, _ in data:
        for f, _ in events:
            vocab.setdefault(f, len(vocab))
    split = len(data) * 4 // 5
    return data[:split], data[split:], vocab


class LstmAttrib(nn.Module):
    def __init__(self, vocab_size, n_objects=16):
        super().__init__()
        self.func_emb = nn.Embedding(vocab_size, HIDDEN)
        self.obj_emb = nn.Embedding(n_objects, HIDDEN)
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, batch_first=True, bidirectional=True)
        self.cause_head = nn.Linear(2 * HIDDEN, 1)

    def forward(self, funcs, objs):
        x = self.func_emb(funcs) + self.obj_emb(objs)
        h, _ = self.lstm(x)
        return self.cause_head(h).squeeze(-1)  # [batch, seq] cause logits


def tensors(events, vocab, device):
    funcs = torch.tensor([[vocab.get(f, 0) for f, _ in events]], device=device)
    objs = torch.tensor([[min(o, 15) for _, o in events]], device=device)
    return funcs, objs


def run_seed(seed, train, test, vocab, device):
    torch.manual_seed(seed)
    model = LstmAttrib(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sup = [(e, l) for e, l in train if l and l[1] < len(e) and l[0] != l[1]]
    for _ in range(EPOCHS):
        for events, (_, cause) in sup:
            funcs, objs = tensors(events, vocab, device)
            logits = model(funcs, objs)
            loss = nn.functional.cross_entropy(
                logits, torch.tensor([cause], device=device)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

    hit1 = hit3 = mrr = n = 0
    with torch.no_grad():
        for events, label in test:
            if not label or label[1] >= len(events) or label[0] == label[1]:
                continue
            crash, cause = label
            funcs, objs = tensors(events, vocab, device)
            logits = model(funcs, objs)[0]
            ranked = [
                s for s in torch.argsort(logits, descending=True).tolist() if s != crash
            ]
            rank = ranked.index(cause) + 1
            hit1 += rank == 1
            hit3 += rank <= 3
            mrr += 1.0 / rank
            n += 1
    return hit1 / n, hit3 / n, mrr / n


def main():
    train, test, vocab = load(sys.argv[1])
    device = "cpu"
    results = [run_seed(s, train, test, vocab, device) for s in SEEDS]
    for i, name in enumerate(["lstm_hit1", "lstm_hit3", "lstm_mrr"]):
        xs = [r[i] for r in results]
        std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        print(f"{name:<12} {statistics.mean(xs):.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
