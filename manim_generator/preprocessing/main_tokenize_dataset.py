from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List

# --- Dépendances (tokenizers HF) ---
try:
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import TemplateProcessing
except ImportError as e:
    raise SystemExit(
        "La lib 'tokenizers' n'est pas installée.\n"
        "Installe-la avec:\n"
        "  pip install tokenizers\n"
    ) from e


# -----------------------------
# Configuration
# -----------------------------
INPUT_PKL = "dataset_dsl_manim.pkl"
OUTPUT_PKL = "dataset_dsl_manim_tokenized.pkl"

TOKENIZER_DIR = "tokenizers"
DSL_TOKENIZER_JSON = os.path.join(TOKENIZER_DIR, "dsl_tokenizer.json")
CODE_TOKENIZER_JSON = os.path.join(TOKENIZER_DIR, "code_tokenizer.json")

# Token spéciaux
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

# Tailles vocab (à ajuster si besoin)
DSL_VOCAB_SIZE = 2000
CODE_VOCAB_SIZE = 8000

# Fréquence minimale pour garder un token
MIN_FREQUENCY = 2


# -----------------------------
# Utilitaires
# -----------------------------
def _load_pairs(path: str) -> List[Dict[str, str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError("Le fichier pickle ne contient pas une liste.")
    # validation légère
    for i, item in enumerate(data[:5]):
        if not isinstance(item, dict) or "dsl" not in item or "code" not in item:
            raise ValueError(f"Entrée {i} invalide: attendu dict avec clés 'dsl' et 'code'.")
    return data


def _train_byte_bpe(texts: List[str], vocab_size: int) -> ByteLevelBPETokenizer:
    """
    Entraîne un Byte-Level BPE tokenizer (style GPT-2) sur une liste de strings.
    """
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Récupérer les ids des tokens spéciaux
    bos_id = tokenizer.token_to_id(BOS)
    eos_id = tokenizer.token_to_id(EOS)
    if bos_id is None or eos_id is None:
        raise RuntimeError("Impossible de récupérer les ids de <bos>/<eos> après entraînement.")

    # Ajoute automatiquement BOS/EOS à chaque séquence encodée
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS} $A {EOS}",
        pair=f"{BOS} $A {EOS} $B:1 {EOS}:1",
        special_tokens=[(BOS, bos_id), (EOS, eos_id)],
    )

    return tokenizer


def _encode_dataset(
    pairs: List[Dict[str, str]],
    dsl_tok: ByteLevelBPETokenizer,
    code_tok: ByteLevelBPETokenizer,
) -> List[Dict[str, Any]]:
    """
    Transforme [{"dsl":..., "code":...}, ...] en
    [{"dsl_ids":[...], "code_ids":[...]}, ...]
    """
    out: List[Dict[str, Any]] = []
    total = len(pairs)

    for i, p in enumerate(pairs):
        dsl = p["dsl"]
        code = p["code"]

        dsl_ids = dsl_tok.encode(dsl).ids
        code_ids = code_tok.encode(code).ids

        out.append(
            {
                "dsl_ids": dsl_ids,
                "code_ids": code_ids,
            }
        )

        # petit log toutes les 1000 lignes
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"Tokenisation: {i+1}/{total}")

    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not os.path.exists(INPUT_PKL):
        raise SystemExit(
            f"Fichier introuvable: {INPUT_PKL}\n"
            "Assure-toi d'être à la racine du projet et d'avoir généré le dataset .pkl."
        )

    print(f"Chargement: {INPUT_PKL}")
    pairs = _load_pairs(INPUT_PKL)
    print(f"OK: {len(pairs)} paires chargées")

    # Corpus séparés
    dsl_texts = [p["dsl"] for p in pairs]
    code_texts = [p["code"] for p in pairs]

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print("\nEntraînement tokenizer DSL (Byte-Level BPE)...")
    dsl_tok = _train_byte_bpe(dsl_texts, vocab_size=DSL_VOCAB_SIZE)
    dsl_tok.save(DSL_TOKENIZER_JSON)
    print(f"Tokenizer DSL sauvegardé: {DSL_TOKENIZER_JSON}")

    print("\nEntraînement tokenizer CODE (Byte-Level BPE)...")
    code_tok = _train_byte_bpe(code_texts, vocab_size=CODE_VOCAB_SIZE)
    code_tok.save(CODE_TOKENIZER_JSON)
    print(f"Tokenizer CODE sauvegardé: {CODE_TOKENIZER_JSON}")

    print("\nTokenisation du dataset...")
    tokenized_pairs = _encode_dataset(pairs, dsl_tok, code_tok)

    print(f"\nSauvegarde dataset tokenisé: {OUTPUT_PKL}")
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(tokenized_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Terminé ✅")
    print(f"- Dataset tokenisé: {OUTPUT_PKL}")
    print(f"- Tokenizer DSL:   {DSL_TOKENIZER_JSON}")
    print(f"- Tokenizer CODE:  {CODE_TOKENIZER_JSON}")


if __name__ == "__main__":
    main()
