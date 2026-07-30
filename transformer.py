"""
Transformer "decoder-only" (façon mini-GPT) implémenté FROM SCRATCH en
NumPy : forward pass, backward pass (rétropropagation manuelle, sans
autodiff) et entraînement par descente de gradient.

Composants implémentés à la main :
- Embeddings de tokens + embeddings positionnels appris
- Self-attention multi-têtes AVEC masque causal (comme dans GPT)
- Connexions résiduelles + LayerNorm (architecture post-LN, comme dans
  l'article original "Attention Is All You Need")
- Feed-forward (Linear -> ReLU -> Linear)
- Projection finale vers le vocabulaire + softmax + cross-entropy

Tâche de démonstration : modèle de langage causal minimal qui apprend à
prédire le token suivant sur une tâche jouet (copie décalée d'une
séquence), pour vérifier que tout le pipeline forward/backward est
correct.

NB : pour rester lisible, tout est calculé pour une seule séquence à la
fois (pas de dimension batch) ; x a la forme (T, D).
"""

import numpy as np

np.random.seed(0)


# ----------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


# ----------------------------------------------------------------------
# Couches de base, chacune avec forward() et backward()
# ----------------------------------------------------------------------

class Linear:
    def __init__(self, d_in, d_out):
        self.W = np.random.randn(d_in, d_out) * (1.0 / np.sqrt(d_in))
        self.b = np.zeros(d_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW += self.x.T @ dout
        self.db += dout.sum(axis=0)
        return dout @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]

    def zero_grad(self):
        self.dW[...] = 0
        self.db[...] = 0


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.dgamma = np.zeros(dim)
        self.dbeta = np.zeros(dim)
        self.eps = eps
        self.cache = None

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        out = self.gamma * xhat + self.beta
        self.cache = (xhat, var)
        return out

    def backward(self, dout):
        xhat, var = self.cache
        N = dout.shape[-1]
        self.dgamma += (dout * xhat).sum(axis=0)
        self.dbeta += dout.sum(axis=0)

        dxhat = dout * self.gamma
        std_inv = 1.0 / np.sqrt(var + self.eps)
        # formule standard de backward de LayerNorm
        dx = (1.0 / N) * std_inv * (
            N * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )
        return dx

    def params_and_grads(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

    def zero_grad(self):
        self.dgamma[...] = 0
        self.dbeta[...] = 0


class MultiHeadSelfAttention:
    """Self-attention multi-têtes avec masque causal."""

    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        self.cache = None

    def _split_heads(self, x):
        T, D = x.shape
        return x.reshape(T, self.n_heads, self.d_head).transpose(1, 0, 2)  # (H,T,Dh)

    def _merge_heads(self, x):
        H, T, Dh = x.shape
        return x.transpose(1, 0, 2).reshape(T, H * Dh)

    def forward(self, x):
        T, D = x.shape
        Q = self._split_heads(self.Wq.forward(x))   # (H,T,Dh)
        K = self._split_heads(self.Wk.forward(x))
        V = self._split_heads(self.Wv.forward(x))

        scale = 1.0 / np.sqrt(self.d_head)
        scores = np.einsum("htd,hsd->hts", Q, K) * scale   # (H,T,T)

        causal_mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores = np.where(causal_mask[None, :, :], -1e9, scores)

        attn = softmax(scores, axis=-1)                      # (H,T,T)
        out_heads = np.einsum("hts,hsd->htd", attn, V)        # (H,T,Dh)
        out = self._merge_heads(out_heads)                    # (T,D)
        out = self.Wo.forward(out)

        self.cache = (Q, K, V, attn, scale, T)
        return out

    def backward(self, dout):
        Q, K, V, attn, scale, T = self.cache

        d_merged = self.Wo.backward(dout)                     # (T,D)
        d_out_heads = self._split_heads(d_merged)              # (H,T,Dh)

        # out_heads = attn @ V
        d_attn = np.einsum("htd,hsd->hts", d_out_heads, V)      # (H,T,T)
        d_V = np.einsum("hts,htd->hsd", attn, d_out_heads)      # (H,T,Dh)

        # softmax backward (par ligne, sur le dernier axe)
        d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))

        # scores = Q K^T * scale
        d_Q = np.einsum("hts,hsd->htd", d_scores, K) * scale
        d_K = np.einsum("hts,htd->hsd", d_scores, Q) * scale

        dQ_flat = self._merge_heads(d_Q)
        dK_flat = self._merge_heads(d_K)
        dV_flat = self._merge_heads(d_V)

        dx = self.Wq.backward(dQ_flat) + self.Wk.backward(dK_flat) + self.Wv.backward(dV_flat)
        return dx

    def params_and_grads(self):
        out = []
        for lin in (self.Wq, self.Wk, self.Wv, self.Wo):
            out += lin.params_and_grads()
        return out

    def zero_grad(self):
        for lin in (self.Wq, self.Wk, self.Wv, self.Wo):
            lin.zero_grad()


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.relu_mask = None

    def forward(self, x):
        h = self.fc1.forward(x)
        self.relu_mask = (h > 0)
        h = relu(h)
        return self.fc2.forward(h)

    def backward(self, dout):
        dh = self.fc2.backward(dout)
        dh = dh * self.relu_mask
        return self.fc1.backward(dh)

    def params_and_grads(self):
        return self.fc1.params_and_grads() + self.fc2.params_and_grads()

    def zero_grad(self):
        self.fc1.zero_grad()
        self.fc2.zero_grad()


class TransformerBlock:
    """Bloc post-LN : x = LN(x + Sublayer(x))  (comme l'article original)."""

    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)
        self.cache = None

    def forward(self, x):
        attn_out = self.attn.forward(x)
        x1 = self.ln1.forward(x + attn_out)

        ff_out = self.ff.forward(x1)
        x2 = self.ln2.forward(x1 + ff_out)

        self.cache = (x, attn_out, x1, ff_out)
        return x2

    def backward(self, dout):
        x, attn_out, x1, ff_out = self.cache

        d_sum2 = self.ln2.backward(dout)          # grad par rapport à (x1 + ff_out)
        d_ff_out = d_sum2
        d_x1_a = d_sum2

        d_x1_b = self.ff.backward(d_ff_out)
        d_x1 = d_x1_a + d_x1_b

        d_sum1 = self.ln1.backward(d_x1)           # grad par rapport à (x + attn_out)
        d_attn_out = d_sum1
        d_x_a = d_sum1

        d_x_b = self.attn.backward(d_attn_out)
        dx = d_x_a + d_x_b
        return dx

    def params_and_grads(self):
        return (self.attn.params_and_grads() + self.ln1.params_and_grads()
                + self.ff.params_and_grads() + self.ln2.params_and_grads())

    def zero_grad(self):
        self.attn.zero_grad()
        self.ln1.zero_grad()
        self.ff.zero_grad()
        self.ln2.zero_grad()


# ----------------------------------------------------------------------
# Modèle complet : Transformer decoder-only (mini-GPT)
# ----------------------------------------------------------------------

class TinyTransformerLM:
    def __init__(self, vocab_size, d_model=32, n_heads=4, d_ff=64,
                 n_layers=2, max_len=64):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.tok_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_len, d_model) * 0.02
        self.d_tok_emb = np.zeros_like(self.tok_emb)
        self.d_pos_emb = np.zeros_like(self.pos_emb)

        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.head = Linear(d_model, vocab_size)   # projection finale vers le vocabulaire

        self.cache = None

    # ---------------- forward ----------------
    def forward(self, token_ids):
        T = len(token_ids)
        x = self.tok_emb[token_ids] + self.pos_emb[:T]
        block_inputs = [x]
        for block in self.blocks:
            x = block.forward(x)
        logits = self.head.forward(x)          # (T, V)
        probs = softmax(logits, axis=-1)
        self.cache = (token_ids, T)
        return probs

    def loss(self, probs, targets):
        eps = 1e-12
        T = len(targets)
        return -np.mean([np.log(probs[t, targets[t]] + eps) for t in range(T)])

    # ---------------- backward ----------------
    def backward(self, probs, targets):
        token_ids, T = self.cache
        V = self.vocab_size

        dlogits = probs.copy()
        for t in range(T):
            dlogits[t, targets[t]] -= 1.0
        dlogits /= T

        dx = self.head.backward(dlogits)        # (T, d_model)

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        # embeddings
        for t in range(T):
            self.d_tok_emb[token_ids[t]] += dx[t]
            self.d_pos_emb[t] += dx[t]

    # ---------------- optimisation ----------------
    def all_params_and_grads(self):
        out = [(self.tok_emb, self.d_tok_emb), (self.pos_emb, self.d_pos_emb)]
        for block in self.blocks:
            out += block.params_and_grads()
        out += self.head.params_and_grads()
        return out

    def zero_grad(self):
        self.d_tok_emb[...] = 0
        self.d_pos_emb[...] = 0
        for block in self.blocks:
            block.zero_grad()
        self.head.zero_grad()

    def step(self, lr, clip=5.0):
        for p, g in self.all_params_and_grads():
            np.clip(g, -clip, clip, out=g)
            p -= lr * g

    def train_step(self, token_ids, targets, lr):
        self.zero_grad()
        probs = self.forward(token_ids)
        loss = self.loss(probs, targets)
        self.backward(probs, targets)
        self.step(lr)
        return loss

    def predict(self, token_ids):
        probs = self.forward(token_ids)
        return [int(np.argmax(probs[t])) for t in range(len(token_ids))]


# ----------------------------------------------------------------------
# Démo : modèle de langage causal sur une tâche jouet
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Tâche : prédire, pour chaque position t, le token à t+1
    # (comme un vrai LM causal) sur des séquences "montantes" a,a+1,a+2,...
    VOCAB_SIZE = 10
    SEQ_LEN = 6

    model = TinyTransformerLM(vocab_size=VOCAB_SIZE, d_model=32, n_heads=4,
                               d_ff=64, n_layers=2, max_len=SEQ_LEN)

    def make_example():
        start = np.random.randint(0, VOCAB_SIZE - SEQ_LEN)
        seq = list(range(start, start + SEQ_LEN + 1))   # longueur SEQ_LEN+1
        x = seq[:-1]        # entrée
        y = seq[1:]         # cible = entrée décalée de 1 (next-token prediction)
        return x, y

    print("Entraînement (prédiction du prochain token)...")
    losses = []
    lr = 0.05
    for epoch in range(4000):
        x, y = make_example()
        loss = model.train_step(x, y, lr=lr)
        losses.append(loss)
        if epoch % 500 == 0:
            print(f"epoch {epoch:5d}  loss={np.mean(losses[-100:]):.4f}")

    print("\nTest après entraînement :")
    correct, total = 0, 0
    for _ in range(10):
        x, y = make_example()
        pred = model.predict(x)
        correct += sum(p == t for p, t in zip(pred, y))
        total += len(y)
        print(f"entrée={x}  cible={y}  prédiction={pred}")
    print(f"\nExactitude par token : {correct/total:.2%}")
