"""
RNN encodeur-décodeur avec mécanisme d'attention (Bahdanau), implémenté
from scratch en NumPy : forward pass, backward pass (BPTT) et descente de
gradient, sans aucun framework de deep learning.

Architecture :
- Encodeur : RNN vanilla (tanh) qui lit la séquence d'entrée et produit
  un état caché à chaque pas de temps.
- Attention : pour chaque pas du décodeur, on calcule un score entre
  l'état du décodeur et chaque état de l'encodeur (score additif),
  on normalise avec un softmax, puis on fait une somme pondérée des
  états de l'encodeur -> "contexte".
- Décodeur : RNN vanilla qui reçoit à chaque pas [embedding(y_prev) ; contexte]
  et produit une distribution de probabilité sur le vocabulaire de sortie.

Tâche de démonstration : apprendre à recopier une séquence de symboles
(seq2seq copy task) pour vérifier que le modèle apprend bien.
"""

import numpy as np

np.random.seed(42)


# ----------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def tanh_deriv(tanh_out):
    # dérivée de tanh exprimée à partir de sa propre sortie
    return 1.0 - tanh_out ** 2


# ----------------------------------------------------------------------
# Modèle : RNN + Attention
# ----------------------------------------------------------------------

class RNNWithAttention:
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32, attn_dim=16):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        H, E, A, V = hidden_dim, embed_dim, attn_dim, vocab_size

        def rnd(*shape):
            return np.random.randn(*shape) * 0.05

        # embeddings partagés entrée/sortie
        self.params = {
            "Emb":       rnd(V, E),

            # encodeur
            "Wxh_enc":   rnd(H, E),
            "Whh_enc":   rnd(H, H),
            "bh_enc":    np.zeros((H, 1)),

            # attention (score additif de Bahdanau)
            "Wa":        rnd(A, H),   # projette l'état encodeur
            "Ua":        rnd(A, H),   # projette l'état décodeur
            "va":        rnd(1, A),

            # décodeur : entrée = [embed(y_prev) ; contexte] -> dim E+H
            "Wxh_dec":   rnd(H, E + H),
            "Whh_dec":   rnd(H, H),
            "bh_dec":    np.zeros((H, 1)),

            # projection de sortie
            "Why":       rnd(V, H),
            "by":        np.zeros((V, 1)),
        }

    # --------------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------------
    def forward(self, x_seq, y_in_seq):
        """
        x_seq     : liste d'indices (entrée, longueur Tx)
        y_in_seq  : liste d'indices donnés en entrée du décodeur à chaque
                    pas (teacher forcing) : typiquement [<SOS>, y1, y2, ...]
        Retourne les probabilités prédites à chaque pas + un cache pour
        le backward.
        """
        p = self.params
        Tx = len(x_seq)
        Ty = len(y_in_seq)
        H = self.hidden_dim

        cache = {"x_seq": x_seq, "y_in_seq": y_in_seq}

        # ---- Encodeur ----
        h_enc = np.zeros((Tx, H, 1))
        x_emb_enc = np.zeros((Tx, self.embed_dim, 1))
        h_prev = np.zeros((H, 1))
        for t in range(Tx):
            x_emb = p["Emb"][x_seq[t]].reshape(-1, 1)
            x_emb_enc[t] = x_emb
            h_t = np.tanh(p["Wxh_enc"] @ x_emb + p["Whh_enc"] @ h_prev + p["bh_enc"])
            h_enc[t] = h_t
            h_prev = h_t
        cache["h_enc"] = h_enc
        cache["x_emb_enc"] = x_emb_enc

        # ---- Décodeur avec attention ----
        s_prev = h_enc[-1].copy()          # état initial du décodeur
        probs = np.zeros((Ty, self.vocab_size, 1))
        dec_cache = []

        for t in range(Ty):
            # scores d'attention e_i = va . tanh(Wa h_i + Ua s_prev)
            proj_enc = np.einsum("ah,thi->tai", p["Wa"], h_enc)   # (Tx, A, 1)
            proj_dec = p["Ua"] @ s_prev                            # (A, 1)
            u = np.tanh(proj_enc + proj_dec[None, :, :])           # (Tx, A, 1)
            e = np.einsum("oa,tai->ti", p["va"], u).reshape(Tx)    # (Tx,)
            alpha = softmax(e)                                     # (Tx,)
            context = np.sum(alpha[:, None, None] * h_enc, axis=0)  # (H,1)

            y_prev_emb = p["Emb"][y_in_seq[t]].reshape(-1, 1)
            dec_in = np.vstack([y_prev_emb, context])              # (E+H,1)

            s_t = np.tanh(p["Wxh_dec"] @ dec_in + p["Whh_dec"] @ s_prev + p["bh_dec"])
            logits = p["Why"] @ s_t + p["by"]
            prob = softmax(logits, axis=0)
            probs[t] = prob

            dec_cache.append({
                "s_prev": s_prev, "s_t": s_t, "alpha": alpha, "u": u,
                "context": context, "dec_in": dec_in, "y_prev_emb": y_prev_emb,
                "prob": prob,
            })
            s_prev = s_t

        cache["dec_cache"] = dec_cache
        return probs, cache

    # --------------------------------------------------------------
    # LOSS (cross-entropy)
    # --------------------------------------------------------------
    def loss(self, probs, target_seq):
        Ty = len(target_seq)
        eps = 1e-12
        total = 0.0
        for t in range(Ty):
            total -= np.log(probs[t, target_seq[t], 0] + eps)
        return total / Ty

    # --------------------------------------------------------------
    # BACKWARD (BPTT complet à travers le décodeur PUIS l'encodeur)
    # --------------------------------------------------------------
    def backward(self, cache, target_seq):
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        x_seq = cache["x_seq"]
        y_in_seq = cache["y_in_seq"]
        h_enc = cache["h_enc"]
        x_emb_enc = cache["x_emb_enc"]
        dec_cache = cache["dec_cache"]
        Tx, H, _ = h_enc.shape
        Ty = len(target_seq)

        d_h_enc_total = np.zeros_like(h_enc)   # gradient accumulé reçu par chaque état encodeur
        d_s_next = np.zeros((H, 1))            # gradient venant du pas décodeur suivant (via Whh_dec)

        for t in reversed(range(Ty)):
            c = dec_cache[t]
            prob = c["prob"].copy()
            # d(loss)/d(logits) pour softmax + cross-entropy
            dlogits = prob.copy()
            dlogits[target_seq[t], 0] -= 1.0
            dlogits /= Ty

            grads["Why"] += dlogits @ c["s_t"].T
            grads["by"] += dlogits

            d_s_t = p["Why"].T @ dlogits + d_s_next     # contribution directe + future
            d_pre_s = d_s_t * tanh_deriv(c["s_t"])        # à travers tanh

            grads["Wxh_dec"] += d_pre_s @ c["dec_in"].T
            grads["Whh_dec"] += d_pre_s @ c["s_prev"].T
            grads["bh_dec"] += d_pre_s

            d_dec_in = p["Wxh_dec"].T @ d_pre_s            # (E+H,1)
            E = self.embed_dim
            d_y_prev_emb = d_dec_in[:E]
            d_context = d_dec_in[E:]

            # gradient vers s_prev (pour le pas précédent du décodeur)
            d_s_prev_from_here = p["Whh_dec"].T @ d_pre_s

            # --- rétropropagation dans le mécanisme d'attention ---
            alpha = c["alpha"]                # (Tx,)
            u = c["u"]                        # (Tx, A, 1)

            # context = sum_i alpha_i * h_enc_i
            d_alpha = np.einsum("hi,thi->t", d_context, h_enc)   # (Tx,)
            for i in range(Tx):
                d_h_enc_total[i] += alpha[i] * d_context

            # softmax backward : d_e = alpha * (d_alpha - sum(alpha*d_alpha))
            s = np.sum(alpha * d_alpha)
            d_e = alpha * (d_alpha - s)        # (Tx,)

            # e_i = va . u_i
            d_va = np.zeros_like(p["va"])
            d_u = np.zeros_like(u)
            for i in range(Tx):
                d_va += d_e[i] * u[i].T
                d_u[i] = (p["va"].T * d_e[i])
            grads["va"] += d_va

            # u_i = tanh(Wa h_i + Ua s_prev)
            d_pre_u = d_u * tanh_deriv(u)      # (Tx, A, 1)

            d_Wa = np.zeros_like(p["Wa"])
            d_Ua = np.zeros_like(p["Ua"])
            d_s_prev_from_attn = np.zeros((H, 1))
            for i in range(Tx):
                d_Wa += d_pre_u[i] @ h_enc[i].T
                d_Ua += d_pre_u[i] @ c["s_prev"].T
                d_h_enc_total[i] += p["Wa"].T @ d_pre_u[i]
                d_s_prev_from_attn += p["Ua"].T @ d_pre_u[i]
            grads["Wa"] += d_Wa
            grads["Ua"] += d_Ua

            # embedding de y_prev
            grads["Emb"][y_in_seq[t]] += d_y_prev_emb.flatten()

            d_s_next = d_s_prev_from_here + d_s_prev_from_attn

        # d_s_next contient maintenant le gradient à injecter dans le
        # dernier état de l'encodeur (s_prev initial du décodeur = h_enc[-1])
        d_h_enc_total[-1] += d_s_next

        # --- BPTT dans l'encodeur ---
        d_h_next = np.zeros((H, 1))
        for t in reversed(range(Tx)):
            d_h = d_h_enc_total[t] + d_h_next
            d_pre_h = d_h * tanh_deriv(h_enc[t])

            grads["Wxh_enc"] += d_pre_h @ x_emb_enc[t].T
            h_prev = h_enc[t - 1] if t > 0 else np.zeros((H, 1))
            grads["Whh_enc"] += d_pre_h @ h_prev.T
            grads["bh_enc"] += d_pre_h

            grads["Emb"][x_seq[t]] += (p["Wxh_enc"].T @ d_pre_h).flatten()

            d_h_next = p["Whh_enc"].T @ d_pre_h

        # clipping pour la stabilité
        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])

        return grads

    # --------------------------------------------------------------
    def update(self, grads, lr):
        for k in self.params:
            self.params[k] -= lr * grads[k]

    def train_step(self, x_seq, y_in_seq, target_seq, lr):
        probs, cache = self.forward(x_seq, y_in_seq)
        loss = self.loss(probs, target_seq)
        grads = self.backward(cache, target_seq)
        self.update(grads, lr)
        return loss

    # --------------------------------------------------------------
    def predict(self, x_seq, sos_id, max_len=10):
        """Génération pas à pas (sans teacher forcing)."""
        p = self.params
        Tx = len(x_seq)
        H = self.hidden_dim

        h_prev = np.zeros((H, 1))
        h_enc = np.zeros((Tx, H, 1))
        for t in range(Tx):
            x_emb = p["Emb"][x_seq[t]].reshape(-1, 1)
            h_prev = np.tanh(p["Wxh_enc"] @ x_emb + p["Whh_enc"] @ h_prev + p["bh_enc"])
            h_enc[t] = h_prev

        s_prev = h_enc[-1].copy()
        y_prev = sos_id
        out = []
        for _ in range(max_len):
            proj_enc = np.einsum("ah,thi->tai", p["Wa"], h_enc)
            proj_dec = p["Ua"] @ s_prev
            u = np.tanh(proj_enc + proj_dec[None, :, :])
            e = np.einsum("oa,tai->ti", p["va"], u).reshape(Tx)
            alpha = softmax(e)
            context = np.sum(alpha[:, None, None] * h_enc, axis=0)

            y_prev_emb = p["Emb"][y_prev].reshape(-1, 1)
            dec_in = np.vstack([y_prev_emb, context])
            s_t = np.tanh(p["Wxh_dec"] @ dec_in + p["Whh_dec"] @ s_prev + p["bh_dec"])
            logits = p["Why"] @ s_t + p["by"]
            prob = softmax(logits, axis=0)
            y_next = int(np.argmax(prob))
            out.append(y_next)
            s_prev = s_t
            y_prev = y_next
        return out


# ----------------------------------------------------------------------
# Démo : tâche "copie de séquence"
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # vocabulaire : symboles 0..4, + <SOS>=5
    VOCAB_SIZE = 6
    SOS = 5
    SEQ_LEN = 5

    model = RNNWithAttention(vocab_size=VOCAB_SIZE, embed_dim=12, hidden_dim=24, attn_dim=12)

    def make_example():
        seq = list(np.random.randint(0, 5, size=SEQ_LEN))
        x_seq = seq
        target_seq = seq                      # la cible = copie de l'entrée
        y_in_seq = [SOS] + seq[:-1]            # teacher forcing
        return x_seq, y_in_seq, target_seq

    print("Entraînement (tâche de copie de séquence)...")
    losses = []
    n_epochs = 10000
    lr = 0.05
    for epoch in range(n_epochs):
        x_seq, y_in_seq, target_seq = make_example()
        loss = model.train_step(x_seq, y_in_seq, target_seq, lr=lr)
        losses.append(loss)
        if epoch % 1000 == 0:
            print(f"epoch {epoch:5d}  loss={np.mean(losses[-200:]):.4f}")

    print("\nTest après entraînement :")
    for _ in range(5):
        x_seq, _, target_seq = make_example()
        pred = model.predict(x_seq, sos_id=SOS, max_len=SEQ_LEN)
        print(f"entrée={x_seq}  cible={target_seq}  prédiction={pred}")
