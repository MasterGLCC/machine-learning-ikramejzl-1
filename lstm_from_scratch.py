"""
LSTM (Long Short-Term Memory) implémenté FROM SCRATCH avec NumPy.

===========================================================================
POURQUOI LE LSTM EXISTE (par rapport au RNN simple)
===========================================================================
Le RNN "vanilla" (voir rnn_from_scratch.py) souffre du problème du
"vanishing gradient" (gradient qui s'évanouit) : sur de longues séquences,
en remontant le temps pendant le BPTT, on multiplie beaucoup de facteurs
< 1 (dérivées de tanh, poids Whh) les uns par les autres. Le gradient
devient exponentiellement petit et le réseau "oublie" les informations
lointaines dans le passé.

Le LSTM résout ce problème en ajoutant :
    1) Un "état de cellule" c_t : une mémoire à long terme qui circule
       presque sans transformation d'un pas de temps à l'autre (juste
       des multiplications élément par élément, pas de matrice), ce qui
       permet au gradient de circuler sur de longues distances.
    2) Trois "portes" (gates) qui contrôlent le flux d'information :
       - la porte d'oubli (forget gate)  : que garder de la mémoire passée ?
       - la porte d'entrée (input gate)  : quoi ajouter à la mémoire ?
       - la porte de sortie (output gate): quoi exposer comme sortie h_t ?

===========================================================================
ÉQUATIONS DU LSTM (à chaque pas de temps t)
===========================================================================
On concatène l'état caché précédent et l'entrée actuelle :
    z_t = [h_{t-1} ; x_t]              shape (hidden_size + input_size, 1)

Puis on calcule les 3 portes et le "candidat" g_t (toutes obtenues par une
transformation linéaire de z_t suivie d'une non-linéarité) :

    f_t = sigmoid(Wf @ z_t + bf)   # porte d'oubli    -> valeurs dans [0, 1]
    i_t = sigmoid(Wi @ z_t + bi)   # porte d'entrée    -> valeurs dans [0, 1]
    o_t = sigmoid(Wo @ z_t + bo)   # porte de sortie   -> valeurs dans [0, 1]
    g_t = tanh(Wc @ z_t + bc)      # candidat mémoire  -> valeurs dans [-1, 1]

Mise à jour de la mémoire à long terme (état de cellule) :
    c_t = f_t * c_{t-1} + i_t * g_t
          ^^^^^^^^^^^^^   ^^^^^^^^^
          ce qu'on garde  ce qu'on ajoute
          du passé        de nouveau

État caché (mémoire à court terme, aussi utilisé comme sortie) :
    h_t = o_t * tanh(c_t)

Sortie du réseau (ex: pour prédire le prochain caractère) :
    y_t = Why @ h_t + by
    p_t = softmax(y_t)

NOTE : "*" désigne ici une multiplication ÉLÉMENT PAR ÉLÉMENT (Hadamard),
PAS un produit matriciel. C'est important : c'est justement parce que la
mémoire c_t circule via des multiplications élément par élément (et pas
des produits matriciels comme Whh dans le RNN simple) que le gradient
peut se propager sur de longues séquences sans s'évanouir aussi vite.

===========================================================================
CONTENU DE CE FICHIER
===========================================================================
    - Forward pass  : calcule gates, cell state, hidden state, sorties
    - Backward pass : BPTT adapté au LSTM (plus de portes = plus de
                       gradients à calculer, mais même principe que pour
                       le RNN simple)
    - Gradient clipping
    - Un exemple d'entraînement : prédiction du caractère suivant
"""

import numpy as np


def sigmoid(x):
    """Fonction sigmoïde : écrase n'importe quelle valeur dans l'intervalle (0, 1).
    Utilisée pour les 3 portes car on veut des valeurs interprétables comme
    des "proportions" (0 = porte fermée, 1 = porte grande ouverte)."""
    return 1.0 / (1.0 + np.exp(-x))


class LSTM:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        """
        input_size  : taille du vecteur d'entrée (ex: taille du vocabulaire)
        hidden_size : taille de l'état caché ET de l'état de cellule
        output_size : taille du vecteur de sortie
        """
        rng = np.random.default_rng(seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # z_t = [h_{t-1} ; x_t] a pour taille (hidden_size + input_size).
        # Chaque porte prend donc z_t en entrée : ses poids ont pour forme
        # (hidden_size, hidden_size + input_size).
        z_size = hidden_size + input_size

        def init_weight(shape):
            # Petite échelle pour démarrer proche de zéro (comme pour le RNN),
            # afin d'éviter de saturer sigmoid/tanh dès le début.
            return rng.standard_normal(shape) * 0.01

        # --- Poids des 4 "portes" (forget, input, output, candidat) ---
        self.Wf = init_weight((hidden_size, z_size))   # porte d'oubli
        self.Wi = init_weight((hidden_size, z_size))   # porte d'entrée
        self.Wo = init_weight((hidden_size, z_size))   # porte de sortie
        self.Wc = init_weight((hidden_size, z_size))   # candidat mémoire (g_t)

        self.bf = np.zeros((hidden_size, 1))
        # Astuce classique : on initialise le biais de la porte d'oubli à 1
        # (plutôt que 0). Au début de l'entraînement, cela pousse f_t proche
        # de sigmoid(1) ≈ 0.73, donc la porte d'oubli "laisse plutôt passer"
        # l'information par défaut. Cela facilite grandement l'apprentissage
        # de dépendances longues dès les premières itérations.
        self.bf += 1.0
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))

        # --- Poids de la couche de sortie (identique au RNN simple) ---
        self.Why = init_weight((output_size, hidden_size))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev, c_prev):
        """
        Propagation avant sur une séquence complète.

        Paramètres
        ----------
        inputs : liste de longueur T de vecteurs one-hot (input_size, 1)
        h_prev : état caché initial (hidden_size, 1)
        c_prev : état de cellule initial (hidden_size, 1)

        Retourne un dictionnaire "cache" contenant TOUTES les valeurs
        intermédiaires à chaque pas de temps (gates, cell states, etc.),
        nécessaires pour le backward pass (BPTT).
        """
        cache = {
            "x": {}, "z": {},
            "f": {}, "i": {}, "o": {}, "g": {},
            "c": {}, "h": {},
            "y": {}, "p": {},
        }
        cache["h"][-1] = np.copy(h_prev)
        cache["c"][-1] = np.copy(c_prev)

        for t in range(len(inputs)):
            x_t = inputs[t]
            h_prev_t = cache["h"][t - 1]
            c_prev_t = cache["c"][t - 1]

            # Concaténation [h_{t-1} ; x_t] -> une seule entrée pour toutes les portes
            z_t = np.vstack((h_prev_t, x_t))

            # --- Les 4 portes ---
            f_t = sigmoid(self.Wf @ z_t + self.bf)   # que garder de c_{t-1} ?
            i_t = sigmoid(self.Wi @ z_t + self.bi)   # combien ajouter de g_t ?
            o_t = sigmoid(self.Wo @ z_t + self.bo)   # combien exposer de c_t ?
            g_t = np.tanh(self.Wc @ z_t + self.bc)   # nouveau contenu candidat

            # --- Mise à jour de la mémoire à long terme ---
            # Multiplication élément par élément (Hadamard), PAS matricielle :
            # c'est ce qui permet au gradient de circuler sans "écrasement"
            # matriciel répété comme dans le RNN simple.
            c_t = f_t * c_prev_t + i_t * g_t

            # --- État caché (sortie exposée à l'extérieur) ---
            h_t = o_t * np.tanh(c_t)

            # --- Couche de sortie (identique au RNN simple) ---
            y_t = self.Why @ h_t + self.by
            exp_y = np.exp(y_t - np.max(y_t))  # stabilité numérique
            p_t = exp_y / np.sum(exp_y)

            # On stocke tout : indispensable pour le backward pass
            cache["x"][t] = x_t
            cache["z"][t] = z_t
            cache["f"][t] = f_t
            cache["i"][t] = i_t
            cache["o"][t] = o_t
            cache["g"][t] = g_t
            cache["c"][t] = c_t
            cache["h"][t] = h_t
            cache["y"][t] = y_t
            cache["p"][t] = p_t

        return cache

    def backward(self, cache, targets):
        """
        Backpropagation Through Time (BPTT) pour le LSTM.

        Le principe général est identique au RNN simple (on remonte le
        temps à l'envers, on accumule les gradients des poids partagés),
        mais il y a maintenant DEUX flux de gradient qui voyagent dans le
        temps au lieu d'un seul :
            - dh_next : gradient qui vient du futur, via l'état caché h_t
            - dc_next : gradient qui vient du futur, via l'état de cellule c_t
        C'est ce deuxième canal (dc_next), qui ne passe PAS par une
        multiplication matricielle mais seulement par f_t (multiplication
        élément par élément), qui est la clé du LSTM : le gradient peut
        traverser de nombreux pas de temps sans être écrasé.
        """
        T = len(cache["x"])

        # Initialisation des gradients accumulés (un par jeu de poids)
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWc = np.zeros_like(self.Wc)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        # Gradients qui circulent dans le temps (nuls au tout dernier pas de temps,
        # car il n'y a pas de "futur" après lui)
        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        loss = 0.0

        for t in reversed(range(T)):
            f_t, i_t, o_t, g_t = cache["f"][t], cache["i"][t], cache["o"][t], cache["g"][t]
            c_t, c_prev_t = cache["c"][t], cache["c"][t - 1]
            h_t, z_t, p_t = cache["h"][t], cache["z"][t], cache["p"][t]

            # --- 1) Perte cross-entropy ---
            loss += -np.log(p_t[targets[t], 0] + 1e-12)

            # --- 2) Gradient softmax + cross-entropy (identique au RNN) ---
            dy = np.copy(p_t)
            dy[targets[t]] -= 1

            # --- 3) Gradients de la couche de sortie ---
            dWhy += dy @ h_t.T
            dby += dy

            # --- 4) Gradient vers l'état caché h_t ---
            # h_t influence : (a) directement y_t, (b) indirectement h_{t+1}
            # ET c_{t+1} (car h_t entre dans z_{t+1} qui alimente TOUTES les
            # portes du pas de temps suivant). dh_next contient déjà cette
            # contribution accumulée du futur.
            dh = self.Why.T @ dy + dh_next

            # --- 5) Gradient vers la porte de sortie o_t ---
            # h_t = o_t * tanh(c_t)  =>  dL/do_t = dh * tanh(c_t)
            do = dh * np.tanh(c_t)
            do_raw = do * o_t * (1 - o_t)   # dérivée de sigmoid : s*(1-s)

            # --- 6) Gradient vers l'état de cellule c_t ---
            # h_t = o_t * tanh(c_t)  =>  contribution directe : dh * o_t * (1 - tanh(c_t)^2)
            # + la contribution qui vient du futur via c_{t+1} = f_{t+1}*c_t + ...
            # (déjà accumulée dans dc_next)
            dc = dh * o_t * (1 - np.tanh(c_t) ** 2) + dc_next

            # --- 7) Gradient vers le candidat g_t ---
            # c_t = f_t * c_{t-1} + i_t * g_t  =>  dL/dg_t = dc * i_t
            dg = dc * i_t
            dg_raw = dg * (1 - g_t ** 2)    # dérivée de tanh

            # --- 8) Gradient vers la porte d'entrée i_t ---
            # c_t = f_t * c_{t-1} + i_t * g_t  =>  dL/di_t = dc * g_t
            di = dc * g_t
            di_raw = di * i_t * (1 - i_t)   # dérivée de sigmoid

            # --- 9) Gradient vers la porte d'oubli f_t ---
            # c_t = f_t * c_{t-1} + i_t * g_t  =>  dL/df_t = dc * c_{t-1}
            df = dc * c_prev_t
            df_raw = df * f_t * (1 - f_t)   # dérivée de sigmoid

            # --- 10) Gradients des poids de chaque porte ---
            # Chaque porte est de la forme sigmoid(W @ z_t + b) ou tanh(W @ z_t + b),
            # donc dW = grad_raw @ z_t^T  et  db = grad_raw (mêmes formules pour les 4)
            dWf += df_raw @ z_t.T
            dWi += di_raw @ z_t.T
            dWo += do_raw @ z_t.T
            dWc += dg_raw @ z_t.T
            dbf += df_raw
            dbi += di_raw
            dbo += do_raw
            dbc += dg_raw

            # --- 11) Gradient vers z_t = [h_{t-1} ; x_t] ---
            # On somme la contribution des 4 portes, car z_t est utilisé
            # par les 4 en parallèle.
            dz = (self.Wf.T @ df_raw + self.Wi.T @ di_raw
                  + self.Wo.T @ do_raw + self.Wc.T @ dg_raw)

            # z_t = [h_{t-1} ; x_t] : les "hidden_size" premières lignes de
            # dz correspondent au gradient par rapport à h_{t-1}. On n'a pas
            # besoin du gradient par rapport à x_t (on ne rétropropage pas
            # jusqu'à l'entrée elle-même dans cet exemple).
            dh_next = dz[: self.hidden_size, :]

            # --- 12) Gradient vers c_{t-1}, à transmettre au pas précédent ---
            # c_t = f_t * c_{t-1} + i_t * g_t  =>  dL/dc_{t-1} = dc * f_t
            # C'est LE chemin clé du LSTM : il ne passe par AUCUNE matrice de
            # poids, seulement par une multiplication élément par élément
            # avec f_t. Si f_t est proche de 1, le gradient traverse presque
            # intact -> plus de problème de vanishing gradient !
            dc_next = dc * f_t

        # --- Gradient clipping (comme pour le RNN simple) ---
        grads = {
            "Wf": dWf, "Wi": dWi, "Wo": dWo, "Wc": dWc,
            "bf": dbf, "bi": dbi, "bo": dbo, "bc": dbc,
            "Why": dWhy, "by": dby,
        }
        for g in grads.values():
            np.clip(g, -5, 5, out=g)

        return loss, grads, cache["h"][T - 1], cache["c"][T - 1]

    def update_params(self, grads, lr=0.1):
        """Descente de gradient stochastique (SGD) sur tous les poids du LSTM."""
        self.Wf -= lr * grads["Wf"]
        self.Wi -= lr * grads["Wi"]
        self.Wo -= lr * grads["Wo"]
        self.Wc -= lr * grads["Wc"]
        self.bf -= lr * grads["bf"]
        self.bi -= lr * grads["bi"]
        self.bo -= lr * grads["bo"]
        self.bc -= lr * grads["bc"]
        self.Why -= lr * grads["Why"]
        self.by -= lr * grads["by"]

    def sample(self, h_prev, c_prev, seed_idx, n, vocab_size):
        """
        Génère une séquence de n indices de façon autoregressive
        (comme pour le RNN simple), mais en propageant maintenant DEUX
        états d'un pas de temps à l'autre : h (mémoire courte) et c (mémoire longue).
        """
        x = np.zeros((vocab_size, 1))
        x[seed_idx] = 1
        h, c = h_prev, c_prev
        indices = []

        for _ in range(n):
            z = np.vstack((h, x))

            f = sigmoid(self.Wf @ z + self.bf)
            i = sigmoid(self.Wi @ z + self.bi)
            o = sigmoid(self.Wo @ z + self.bo)
            g = np.tanh(self.Wc @ z + self.bc)

            c = f * c + i * g
            h = o * np.tanh(c)

            y = self.Why @ h + self.by
            p = np.exp(y - np.max(y))
            p = p / np.sum(p)

            idx = np.random.choice(range(vocab_size), p=p.ravel())
            x = np.zeros((vocab_size, 1))
            x[idx] = 1
            indices.append(idx)

        return indices


# ---------------------------------------------------------------------------
# Exemple d'utilisation : prédiction du caractère suivant dans un texte
# (même tâche que pour le RNN simple, pour pouvoir comparer les deux)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = "bonjour tout le monde bonjour a tous "
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # --- Hyperparamètres ---
    hidden_size = 64
    seq_length = 8
    learning_rate = 0.1
    n_iterations = 2000

    lstm = LSTM(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size)

    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))
    pointer = 0

    for it in range(n_iterations):
        if pointer + seq_length + 1 >= len(data):
            pointer = 0
            h_prev = np.zeros((hidden_size, 1))
            c_prev = np.zeros((hidden_size, 1))

        input_chars = data[pointer: pointer + seq_length]
        target_chars = data[pointer + 1: pointer + seq_length + 1]

        inputs = []
        for ch in input_chars:
            x = np.zeros((vocab_size, 1))
            x[char_to_idx[ch]] = 1
            inputs.append(x)

        targets = [char_to_idx[ch] for ch in target_chars]

        # --- Un pas d'entraînement complet ---
        cache = lstm.forward(inputs, h_prev, c_prev)
        loss, grads, h_prev, c_prev = lstm.backward(cache, targets)
        lstm.update_params(grads, lr=learning_rate)

        if it % 200 == 0:
            print(f"Itération {it:4d} | Loss = {loss:.4f}")

        pointer += seq_length

    # --- Génération de texte après entraînement ---
    print("\n--- Texte généré ---")
    seed_idx = char_to_idx[data[0]]
    sample_idx = lstm.sample(
        np.zeros((hidden_size, 1)), np.zeros((hidden_size, 1)),
        seed_idx, n=50, vocab_size=vocab_size,
    )
    generated_text = "".join(idx_to_char[i] for i in sample_idx)
    print(generated_text)
