"""Zero-shot "is this photo a fungus?" scores for every stored photo vector (CPU only).

Read-only: the manifest is opened with mode=ro (or a scratch copy), the vectors are the
stored BioCLIP-2 image embeddings, and only BioCLIP-2's text tower runs. Writes one npz
per backbone under data/audits/non-fungus-scan/ (private, never committed):
photo_id, probs (photos x classes, float16), classes.

Usage: python zs_score.py <manifest.sqlite> [backbone ...]
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np

from mycomap_vision import config
from mycomap_vision.serving import map_embeddings

OUT = config.DATA_DIR / "audits" / "non-fungus-scan"

# class -> prompts. Kept classes (fungus, slip-label, microscope, habitat) are never junk:
# Steve decided 2026-09-29 and 2026-10-09 not to filter slips, habitat or microscope photos.
CLASSES: dict[str, list[str]] = {
    "fungus": [
        "a photo of a mushroom", "a photo of fungi", "a photo of a mushroom growing on the ground",
        "a photo of a bracket fungus on wood", "a photo of a crust fungus on a log",
        "a photo of a cup fungus", "a photo of a puffball", "a photo of a coral fungus",
        "a photo of a jelly fungus", "a photo of a slime mold", "a photo of a lichen",
        "the gills under a mushroom cap", "a photo of a mushroom cut in half",
        "a photo of mushrooms on a table", "Fungi", "Fungi Basidiomycota Agaricomycetes Agaricales",
        "Fungi Ascomycota Pezizomycetes Pezizales", "Fungi Basidiomycota Agaricomycetes Polyporales",
    ],
    "bird": ["a photo of a bird", "a photo of a duck", "a photo of a goose on water",
             "a photo of a songbird", "Animalia Chordata Aves",
             "Animalia Chordata Aves Anseriformes Anatidae"],
    "mammal": ["a photo of a mammal", "a photo of a dog", "a photo of a cat",
               "a photo of a deer", "a photo of a squirrel", "Animalia Chordata Mammalia"],
    "insect-arthropod": ["a photo of an insect", "a photo of a beetle", "a photo of a butterfly",
                         "a photo of a spider", "a photo of a moth", "a photo of a bee",
                         "Animalia Arthropoda Insecta", "Animalia Arthropoda Arachnida"],
    "other-animal": ["a photo of a frog", "a photo of a snake", "a photo of a salamander",
                     "a photo of a lizard", "a photo of a fish", "a photo of a snail",
                     "Animalia Chordata Amphibia", "Animalia Chordata Reptilia"],
    "person": ["a photo of a person", "a selfie", "a photo of a group of people",
               "a photo of a person's face", "a photo of people hiking"],
    "plant-only": ["a photo of a flower", "a photo of a green plant", "a photo of wildflowers",
                   "a photo of a fern", "a photo of a tree", "a photo of green leaves",
                   "Plantae Tracheophyta Magnoliopsida"],
    "document-screen-map": ["a screenshot of a phone screen", "a screenshot of a web page",
                            "a map", "a printed document with text", "a chart or graph",
                            "a photo of a computer screen", "a page of text"],
    "slip-label": ["a paper label with handwriting next to a mushroom",
                   "a voucher slip with a barcode and a mushroom", "a mushroom next to a ruler",
                   "a handwritten note on paper"],
    "microscope": ["a microscope image of fungal spores", "a micrograph of cells",
                   "a microscope image", "a petri dish with fungal culture",
                   "a photo of a DNA gel"],
    "habitat": ["a photo of a forest", "a photo of a forest floor with leaf litter",
                "a photo of a landscape", "a photo of a grassy field", "a photo of a log in the woods"],
}
JUNK = ("bird", "mammal", "insect-arthropod", "other-animal", "person", "plant-only",
        "document-screen-map")


def text_embeddings() -> tuple[np.ndarray, float]:
    """Unit class vectors (mean of unit prompt vectors, renormalised) and the logit scale."""
    import open_clip
    import torch
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    model.eval()
    out = []
    with torch.inference_mode():
        for prompts in CLASSES.values():
            t = model.encode_text(tok(prompts)).float()
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(0)
            out.append((m / m.norm()).numpy())
    return np.stack(out).astype(np.float32), float(model.logit_scale.exp())


def main() -> None:
    manifest = sys.argv[1]
    backbones = sys.argv[2:] or ["bioclip-2", "bioclip-2-ft-20261007-165400"]
    conn = sqlite3.connect(Path(manifest).resolve().as_uri() + "?mode=ro", uri=True)
    T, scale = text_embeddings()
    print(f"text vectors {T.shape}, logit scale {scale:.1f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for bb in backbones:
        try:
            ids, vecs = map_embeddings(conn, bb)
        except sqlite3.OperationalError:   # read-only: the schema script can't run
            from mycomap_vision.embed import load_embeddings  # noqa: F401
            raise
        n = len(ids)
        probs = np.zeros((n, len(CLASSES)), np.float16)
        step = 50_000
        for a in range(0, n, step):
            v = np.asarray(vecs[a:a + step], dtype=np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            logits = scale * (v @ T.T)
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            probs[a:a + step] = p / p.sum(1, keepdims=True)
            print(f"  {bb}: {min(a + step, n):,}/{n:,}", flush=True)
        np.savez_compressed(OUT / f"zs-{bb}.npz", photo_id=ids.astype(np.int64), probs=probs,
                            classes=np.array(list(CLASSES)), junk=np.array(JUNK),
                            text=T, scale=np.array(scale))
        print(f"{bb}: wrote {n:,} photos", flush=True)


if __name__ == "__main__":
    main()
