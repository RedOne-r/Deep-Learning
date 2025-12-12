import math
import torch
import torch.nn as nn


class attention(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.softmax = nn.Softmax(dim=-1)

    def dot_attention(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embedding_size)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = self.softmax(scores)
        return torch.matmul(attn_weights, v)


def causal_mask(seq_len: int, device, dtype=torch.float32):
    m = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    m = m.masked_fill(m == 1, float("-inf"))
    return m.view(1, 1, seq_len, seq_len)


def key_padding_mask_to_attn_mask(key_padding_mask: torch.Tensor, dtype=torch.float32):
    m = key_padding_mask.to(torch.bool)
    m = m.unsqueeze(1).unsqueeze(2)
    return m.to(dtype=dtype) * float("-inf")


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        assert embedding_size % n_heads == 0
        self.head_dim = embedding_size // n_heads

        self.q_linear = nn.Linear(embedding_size, embedding_size)
        self.k_linear = nn.Linear(embedding_size, embedding_size)
        self.v_linear = nn.Linear(embedding_size, embedding_size)
        self.out_linear = nn.Linear(embedding_size, embedding_size)

        self.attn = attention(self.head_dim)

    def forward(self, q_in, k_in=None, v_in=None, attn_mask=None, key_padding_mask=None):
        if k_in is None:
            k_in = q_in
        if v_in is None:
            v_in = k_in

        B, Tq, _ = q_in.size()
        _, Tk, _ = k_in.size()

        q = self.q_linear(q_in)
        k = self.k_linear(k_in)
        v = self.v_linear(v_in)

        q = q.view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        if key_padding_mask is not None:
            pad_mask = key_padding_mask_to_attn_mask(key_padding_mask, dtype=q.dtype)
            attn_mask = pad_mask if attn_mask is None else (attn_mask + pad_mask)

        out = self.attn.dot_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.embedding_size)
        return self.out_linear(out)


class feedforward(nn.Module):
    def __init__(self, embedding_size: int, ff_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(embedding_size, ff_dim)
        self.layer2 = nn.Linear(ff_dim, embedding_size)
        self.act = nn.GELU()

    def forward(self, x):
        return self.layer2(self.act(self.layer1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.mha = MultiHeadAttention(embedding_size, n_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.ffn = feedforward(embedding_size, ff_dim)

    def forward(self, x, src_key_padding_mask=None):
        a = self.mha(x, attn_mask=None, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + a)
        f = self.ffn(x)
        x = self.norm2(x + f)
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers: int, embedding_size: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embedding_size, n_heads, ff_dim) for _ in range(n_layers)])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_size, n_heads)
        self.cross_attn = MultiHeadAttention(embedding_size, n_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)
        self.ffn = feedforward(embedding_size, ff_dim)

    def forward(self, x, enc_out, tgt_key_padding_mask=None, src_key_padding_mask=None):
        m = causal_mask(x.size(1), x.device, x.dtype)
        a1 = self.self_attn(x, x, x, attn_mask=m, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + a1)
        a2 = self.cross_attn(x, enc_out, enc_out, attn_mask=None, key_padding_mask=src_key_padding_mask)
        x = self.norm2(x + a2)
        f = self.ffn(x)
        x = self.norm3(x + f)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers: int, embedding_size: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_size, n_heads, ff_dim) for _ in range(n_layers)])

    def forward(self, x, enc_out, tgt_key_padding_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(
                x,
                enc_out,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int,
        embedding_size: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        ff_dim: int,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding_size = embedding_size
        self.max_len = max_len

        self.src_tok_emb = nn.Embedding(src_vocab_size, embedding_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, embedding_size)
        self.src_pos_emb = nn.Embedding(max_len, embedding_size)
        self.tgt_pos_emb = nn.Embedding(max_len, embedding_size)

        self.encoder = Encoder(n_encoder_layers, embedding_size, n_heads, ff_dim)
        self.decoder = Decoder(n_decoder_layers, embedding_size, n_heads, ff_dim)

        self.out_proj = nn.Linear(embedding_size, tgt_vocab_size)

    def _add_positional(self, tok_emb, pos_emb, seq_len: int):
        pos = torch.arange(seq_len, device=tok_emb.device).unsqueeze(0)
        return tok_emb + pos_emb(pos)

    def forward(self, src_ids, tgt_ids, src_key_padding_mask=None, tgt_key_padding_mask=None):
        B, S = src_ids.size()
        _, T = tgt_ids.size()

        if S > self.max_len or T > self.max_len:
            raise ValueError(f"seq_len > max_len (got src={S}, tgt={T}, max_len={self.max_len})")

        if src_key_padding_mask is None:
            src_key_padding_mask = (src_ids == self.pad_id)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = (tgt_ids == self.pad_id)

        src = self._add_positional(self.src_tok_emb(src_ids), self.src_pos_emb, S)
        tgt = self._add_positional(self.tgt_tok_emb(tgt_ids), self.tgt_pos_emb, T)

        enc_out = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        dec_out = self.decoder(
            tgt,
            enc_out,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        logits = self.out_proj(dec_out)
        return logits
