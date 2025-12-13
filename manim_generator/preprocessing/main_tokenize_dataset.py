from __future__ import annotations

import pickle
from typing import List, Tuple

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

MAX_LEN = 200  # padding/truncation fixés à 200

# Sorties
OUTPUT_PKL = f"dataset_dsl_manim_tokenized_pad{MAX_LEN}_tuples.pkl"
DSL_TOKENIZER_JSON = f"dsl_tokenizer_pad{MAX_LEN}.json"
CODE_TOKENIZER_JSON = f"code_tokenizer_pad{MAX_LEN}.json"

# Tokens spéciaux
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

# Vocabs (ajustables)
DSL_VOCAB_SIZE = 2000
CODE_VOCAB_SIZE = 8000
MIN_FREQUENCY = 2


# -----------------------------
# Load dataset
# -----------------------------
def load_pairs(path: str) -> List[dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError("Le pickle doit contenir une liste.")

    if data:
        ex = data[0]
        if not isinstance(ex, dict) or "dsl" not in ex or "code" not in ex:
            raise ValueError("Chaque élément doit être un dict avec clés 'dsl' et 'code'.")

    return data


# -----------------------------
# Train + configure tokenizer
# -----------------------------
def train_byte_bpe(texts: List[str], vocab_size: int, max_len: int) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()

    tok.train_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # ids tokens spéciaux
    pad_id = tok.token_to_id(PAD)
    bos_id = tok.token_to_id(BOS)
    eos_id = tok.token_to_id(EOS)
    if pad_id is None or bos_id is None or eos_id is None:
        raise RuntimeError("Impossible de récupérer pad/bos/eos ids après entraînement.")

    # Ajout BOS/EOS automatique
    tok._tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS} $A {EOS}",
        pair=f"{BOS} $A {EOS} $B:1 {EOS}:1",
        special_tokens=[(BOS, bos_id), (EOS, eos_id)],
    )

    # Truncation + padding à longueur fixe
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=max_len, pad_id=pad_id, pad_token=PAD)

    return tok


# -----------------------------
# Encode dataset (compact tuples)
# -----------------------------
def encode_dataset_as_tuples(
    pairs: List[dict],
    dsl_tok: ByteLevelBPETokenizer,
    code_tok: ByteLevelBPETokenizer,
) -> List[Tuple[List[int], List[int]]]:
    """
    Retourne une liste de couples (dsl_ids, code_ids), chacun de longueur MAX_LEN.
    """
    out: List[Tuple[List[int], List[int]]] = []
    total = len(pairs)

    for i, p in enumerate(pairs):
        dsl_ids = dsl_tok.encode(p["dsl"]).ids
        code_ids = code_tok.encode(p["code"]).ids

        out.append((dsl_ids, code_ids))

        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"Encodage: {i+1}/{total}")

    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print(f"Chargement: {INPUT_PKL}")
    pairs = load_pairs(INPUT_PKL)
    print(f"OK: {len(pairs)} paires")

    dsl_texts = [p["dsl"] for p in pairs]
    code_texts = [p["code"] for p in pairs]

    print("\nEntraînement tokenizer DSL (Byte-level BPE)...")
    dsl_tok = train_byte_bpe(dsl_texts, vocab_size=DSL_VOCAB_SIZE, max_len=MAX_LEN)
    dsl_tok._tokenizer.save(DSL_TOKENIZER_JSON)
    print(f"Tokenizer DSL sauvegardé: {DSL_TOKENIZER_JSON}")

    print("\nEntraînement tokenizer CODE (Byte-level BPE)...")
    code_tok = train_byte_bpe(code_texts, vocab_size=CODE_VOCAB_SIZE, max_len=MAX_LEN)
    code_tok._tokenizer.save(CODE_TOKENIZER_JSON)
    print(f"Tokenizer CODE sauvegardé: {CODE_TOKENIZER_JSON}")

    print("\nTokenisation + padding/truncation du dataset (format tuples)...")
    tokenized = encode_dataset_as_tuples(pairs, dsl_tok, code_tok)

    print(f"\nSauvegarde dataset tokenisé: {OUTPUT_PKL}")
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(tokenized, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nTerminé ✅")
    print("Fichiers générés :")
    print("-", OUTPUT_PKL)
    print("-", DSL_TOKENIZER_JSON)
    print("-", CODE_TOKENIZER_JSON)


if __name__ == "__main__":
    main()

