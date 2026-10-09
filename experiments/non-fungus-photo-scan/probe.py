"""A linear probe for "not a fungus" trained on a free labelled set: the mislinked records'
photos are photos of the iNat observation that shares their number, whose iNat taxon is known
(bird, mammal, insect, other animal, plant). Negatives: photos of genuine iNat fungus records.
Three classes on the base BioCLIP-2 vectors (CPU, torch): fungus / animal / plant.

Scores every reference photo; the hand-checked zero-shot sample is never trained on.
Writes data/audits/non-fungus-scan/probe.npz (private).
"""
import json
import pickle
import sqlite3

import numpy as np
import torch

from mycomap_vision import config

from rescore import mislinked
from zs_score import OUT

ANIMAL = {3, 40151, 47158, 1}     # Aves, Mammalia, Insecta, Animalia
PLANT = {47126}
FUNGI = 47170


def main() -> None:
    torch.manual_seed(0)
    F = pickle.load(open(OUT / "frame.pkl", "rb"))
    rows = F["rows"]
    z = np.load(OUT / "zs-bioclip-2.npz")
    row_of = {int(p): i for i, p in enumerate(z["photo_id"].tolist())}
    conn = sqlite3.connect((config.DATA_DIR / "manifest.sqlite").resolve().as_uri() + "?mode=ro",
                           uri=True)
    kingdom = {}
    for oid, anc, tid in conn.execute("select observation_id, taxon_ancestor_ids, taxon_id "
                                      "from inat_observations where status = 'ok'"):
        ids = set(json.loads(anc)) if anc else set()
        if tid:
            ids.add(tid)
        kingdom[oid] = (0 if FUNGI in ids else 2 if ids & PLANT else 1 if ids & ANIMAL else -1)
    bad = mislinked()
    held = set(json.load(open(OUT / "check-inat.json"))["rows"])
    from mycomap_vision.serving import map_embeddings
    mconn = sqlite3.connect((config.DATA_DIR / "manifest.sqlite").resolve().as_uri() + "?mode=ro",
                            uri=True)
    ids, vecs = map_embeddings(mconn, "bioclip-2")
    assert np.array_equal(ids, z["photo_id"])
    X = np.asarray(vecs[np.array([row_of[int(p)] for p in F["photo_id"]])], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.full(len(rows), -1)
    rng = np.random.default_rng(0)
    for i, r in enumerate(rows):
        if r[0] in bad and kingdom.get(r[0], -1) in (1, 2):
            y[i] = kingdom[r[0]]
        elif r[0] not in bad and kingdom.get(r[0], -1) == 0 and i not in held and rng.random() < 0.12:
            y[i] = 0
    tr = np.flatnonzero(y >= 0)
    print("training photos:", {c: int((y[tr] == c).sum()) for c in (0, 1, 2)}, flush=True)
    Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr])
    w = torch.tensor([1.0, (yt == 0).sum() / (yt == 1).sum(), (yt == 0).sum() / (yt == 2).sum()])
    lin = torch.nn.Linear(X.shape[1], 3)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = torch.nn.CrossEntropyLoss(weight=w.float())
    for epoch in range(300):
        opt.zero_grad()
        loss = lossf(lin(Xt * 10), yt)
        loss.backward()
        opt.step()
    print(f"final loss {loss.item():.4f}", flush=True)
    with torch.no_grad():
        P = torch.softmax(lin(torch.tensor(X) * 10), 1).numpy()
    np.savez_compressed(OUT / "probe.npz", photo_id=F["photo_id"], probs=P,
                        trained=np.isin(np.arange(len(rows)), tr))


if __name__ == "__main__":
    main()
