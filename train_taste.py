#!/usr/bin/env python3
"""Train an HT-specific taste score from pairwise picks.

Model: a linear score s(x) = w . x on the stored ViT-L/14 image embeddings,
fitted Bradley-Terry style with L2 regularisation:
  - "a" / "b" picks:  P(winner beats loser) = sigmoid(s(winner) - s(loser))
  - "both_bad":       each photo falls below a learned floor t:
                      P(below) = sigmoid(t - s(x))
  - "tie":            skipped (it carries little signal at this scale)

Evaluation, in order of trust:
  1. Held-out tier labels (calibration/casey-labels.json): 30 photos Casey
     sorted into hero / secondary / reject on 23 Sep, never used in training.
     Spearman of the score against the tiers. LAION scored 0.14 here.
  2. Pairwise accuracy under 5-fold cross-validation on the picks themselves.

Usage:
    python train_taste.py --votes votes.json [--write]

--votes is a JSON list of {choice, a, b} records (exported from the artifact
database). --write stores the score for every photo in the taste_score column
and the weights in models/taste-linear.npz.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

CAL = Path.home() / "Documents/Projects/HimalayanTrust/dev/photo-library/calibration"
TIERS = {"reject": 0, "secondary": 1, "hero": 2}


def spearman(x, y):
    def rank(a):
        a = np.asarray(a, float)
        order = a.argsort()
        r = np.empty(len(a))
        s = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and s[j + 1] == s[i]:
                j += 1
            r[order[i : j + 1]] = (i + j) / 2
            i = j + 1
        return r

    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def fit(X, wins, bads, l2, epochs=400):
    """wins: list of (winner_row, loser_row); bads: list of rows below floor."""
    Xt = torch.tensor(X, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, t], max_iter=epochs, line_search_fn="strong_wolfe")
    W = torch.tensor(wins, dtype=torch.long) if wins else None
    B = torch.tensor(bads, dtype=torch.long) if bads else None

    def closure():
        opt.zero_grad()
        s = Xt @ w
        loss = torch.zeros(())
        n = 0
        if W is not None:
            loss = loss + torch.nn.functional.softplus(-(s[W[:, 0]] - s[W[:, 1]])).sum()
            n += len(W)
        if B is not None:
            loss = loss + torch.nn.functional.softplus(-(t - s[B])).sum()
            n += len(B)
        loss = loss / max(n, 1) + l2 * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(t.detach())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=Path("data-l14/metadata.db"))
    ap.add_argument("--l2", type=float, default=None, help="Fix the regulariser instead of choosing it by CV")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    votes = json.loads(args.votes.read_text())
    con = sqlite3.connect(args.db)
    need = {v["a"] for v in votes} | {v["b"] for v in votes}
    key = {k["n"]: k["image_id"] for k in json.loads((CAL / "photo-calibration-key.json").read_text())}
    labels = json.loads((CAL / "casey-labels.json").read_text())["labels"]
    held = {key[int(n)]: TIERS[t] for n, t in labels.items()}
    need |= set(held)

    ids, vecs, laion = [], [], {}
    for iid, blob, aes in con.execute(
        f"SELECT image_id, embedding_vector, aesthetic_score FROM images WHERE image_id IN ({','.join('?' * len(need))})",
        list(need),
    ):
        ids.append(iid)
        vecs.append(np.frombuffer(blob, dtype=np.float32))
        laion[iid] = aes
    row = {iid: i for i, iid in enumerate(ids)}
    X = np.array(vecs)

    decisive = [v for v in votes if v["choice"] in ("a", "b")]
    wins = [(row[v[v["choice"]]], row[v["b" if v["choice"] == "a" else "a"]]) for v in decisive]
    bads = [row[v[s]] for v in votes if v["choice"] == "both_bad" for s in ("a", "b")]
    counts = {c: sum(v["choice"] == c for v in votes) for c in ("a", "b", "tie", "both_bad")}
    print(f"votes: {len(votes)}  {counts}")

    # Choose the regulariser by 5-fold CV pairwise accuracy.
    rng = np.random.default_rng(0)
    folds = rng.permutation(len(wins)) % 5
    grid = [args.l2] if args.l2 is not None else [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    best = None
    for l2 in grid:
        acc = []
        for f in range(5):
            tr = [wins[i] for i in range(len(wins)) if folds[i] != f]
            te = [wins[i] for i in range(len(wins)) if folds[i] == f]
            if not te:
                continue
            w, _ = fit(X, tr, bads, l2)
            s = X @ w
            acc.append(np.mean([s[a] > s[b] for a, b in te]))
        m = float(np.mean(acc))
        print(f"  l2={l2:g}  CV pairwise accuracy {m:.3f}")
        if best is None or m > best[1]:
            best = (l2, m)
    l2 = best[0]

    w, t = fit(X, wins, bads, l2)
    hx = [row[i] for i in held]
    s_held = X[hx] @ w
    tiers = list(held.values())
    rho_taste = spearman(s_held, tiers)
    rho_laion = spearman([laion[i] for i in held], tiers)
    print(f"\nchosen l2={l2:g}  CV accuracy {best[1]:.3f}")
    print(f"HELD-OUT tiers (n={len(held)}): taste rho {rho_taste:.2f}   LAION rho {rho_laion:.2f}   bar 0.50")
    for name, v in TIERS.items():
        sel = [s for s, tv in zip(s_held, tiers) if tv == v]
        print(f"  {name:9s} mean taste {np.mean(sel):+.3f}  (n={len(sel)})")

    if args.write:
        allrows = con.execute("SELECT image_id, embedding_vector FROM images WHERE embedding_vector IS NOT NULL").fetchall()
        A = np.array([np.frombuffer(b, dtype=np.float32) for _, b in allrows])
        scores = A @ w
        mu, sd = float(scores.mean()), float(scores.std())
        cols = {r[1] for r in con.execute("PRAGMA table_info(images)")}
        if "taste_score" not in cols:
            con.execute("ALTER TABLE images ADD COLUMN taste_score REAL")
        con.executemany(
            "UPDATE images SET taste_score = ? WHERE image_id = ?",
            [(float((sc - mu) / sd), iid) for sc, (iid, _) in zip(scores, allrows)],
        )
        con.execute(
            "INSERT OR REPLACE INTO index_metadata (key, value) VALUES ('taste_model', ?)",
            (json.dumps({"votes": len(votes), "l2": l2, "cv_acc": best[1], "heldout_rho": rho_taste,
                         "floor_z": (t - mu) / sd}),),
        )
        con.commit()
        Path("models").mkdir(exist_ok=True)
        np.savez("models/taste-linear.npz", w=w, t=t, mu=mu, sd=sd)
        print(f"wrote taste_score (z-scored) for {len(allrows)} photos; floor at z={(t - mu) / sd:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
