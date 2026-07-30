"""
RNN (Recurrent Neural Network) implémenté FROM SCRATCH avec NumPy (aucun framework de deep learning).

===========================================================================
IDÉE GÉNÉRALE
===========================================================================
Un RNN traite une séquence (ex: une phrase, une série temporelle) un élément
à la fois, en gardant en mémoire un "état caché" (hidden state) qui résume
tout ce qu'il a vu jusqu'ici. C'est ce qui lui permet de modéliser des
dépendances temporelles (le mot suivant dépend des mots précédents, etc.)

À chaque pas de temps t, deux équations :

    h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)   <-- met à jour la mémoire
    y_t = Why @ h_t + by                          <-- produit une sortie

Avec :
    x_t : entrée au temps t                (input_size, 1)
    h_t : état caché au temps t            (hidden_size, 1)
    y_t : sortie brute (logits) au temps t (output_size, 1)

    Wxh : poids entrée -> caché            (hidden_size, input_size)
    Whh : poids caché -> caché (récurrent) (hidden_size, hidden_size)
    Why : poids caché -> sortie            (output_size, hidden_size)
    bh, by : biais

Le MÊME jeu de poids (Wxh, Whh, Why) est réutilisé à CHAQUE pas de temps.
C'est le "partage de paramètres" qui permet au RNN de traiter des séquences
de longueur variable.

===========================================================================
CONTENU DE CE FICHIER
===========================================================================
    - Forward pass  : calcule les prédictions du réseau pour une séquence
    - Backward pass : Backpropagation Through Time (BPTT), calcule les
                       gradients de la perte par rapport à tous les poids
    - Gradient clipping : évite l'explosion des gradients (problème
                       classique des RNN sur les séquences longues)
    - Un exemple d'entraînement : prédire le caractère suivant dans un texte
"""

import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        """
        Initialise les poids du réseau.

        input_size  : taille du vecteur d'entrée (ex: taille du vocabulaire
                      si on encode les caractères en one-hot)
        hidden_size : nombre de neurones dans la couche cachée (la "mémoire"
                      du réseau). Plus il est grand, plus le réseau peut
                      mémoriser d'informations, mais plus il est coûteux
                      à entraîner.
        output_size : taille du vecteur de sortie (souvent = input_size
                      quand on prédit le prochain élément de la séquence)
        seed        : graine aléatoire, pour des résultats reproductibles
        """
        rng = np.random.default_rng(seed)

        self.hidden_size = hidden_size

        # --- Initialisation des poids ---
        # On multiplie par 0.01 pour démarrer avec des poids petits :
        # cela évite que tanh() sature dès le début (ce qui bloquerait
        # l'apprentissage à cause de gradients quasi nuls).
        self.Wxh = rng.standard_normal((hidden_size, input_size)) * 0.01   # entrée -> caché
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) * 0.01  # caché -> caché (récurrence)
        self.Why = rng.standard_normal((output_size, hidden_size)) * 0.01  # caché -> sortie

        # Les biais peuvent démarrer à zéro sans problème (contrairement aux poids,
        # ils n'ont pas de risque de "symétrie" entre neurones).
        self.bh = np.zeros((hidden_size, 1))  # biais de la couche cachée
        self.by = np.zeros((output_size, 1))  # biais de la couche de sortie

    def forward(self, inputs, h_prev):
        """
        Propagation avant (forward pass) sur une séquence complète.

        Paramètres
        ----------
        inputs : liste de longueur T, chaque élément est un vecteur one-hot
                 de shape (input_size, 1). Représente la séquence d'entrée
                 (ex: les caractères "b", "o", "n", ...).
        h_prev : état caché initial, shape (hidden_size, 1).
                 En général un vecteur de zéros au début de l'entraînement,
                 ou l'état caché final du batch précédent (pour préserver
                 la mémoire entre deux appels consécutifs).

        Retourne
        --------
        xs : dict {t: x_t}   -> entrées, pour les réutiliser dans le backward
        hs : dict {t: h_t}   -> états cachés à chaque pas de temps
                                 (hs[-1] = h_prev, pratique pour t=0)
        ys : dict {t: y_t}   -> logits (sortie brute avant softmax)
        ps : dict {t: p_t}   -> probabilités (après softmax)

        On garde TOUT en mémoire (et pas seulement le dernier résultat) car
        le backward pass (BPTT) a besoin de x_t et h_t à CHAQUE pas de temps
        pour calculer les gradients.
        """
        xs, hs, ys, ps = {}, {}, {}, {}

        # hs[-1] joue le rôle de "h_{t-1}" quand t=0 (astuce d'indexation Python)
        hs[-1] = np.copy(h_prev)

        for t in range(len(inputs)):
            xs[t] = inputs[t]

            # --- Mise à jour de l'état caché ---
            # Combine l'information de l'entrée actuelle (Wxh @ x_t) avec
            # la mémoire du passé (Whh @ h_{t-1}), puis applique tanh pour
            # borner les valeurs entre -1 et 1 (stabilité + non-linéarité).
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t - 1] + self.bh)

            # --- Calcul de la sortie brute (logits) ---
            # Simple transformation linéaire de l'état caché.
            ys[t] = self.Why @ hs[t] + self.by

            # --- Softmax : transforme les logits en probabilités ---
            # On soustrait le max avant l'exponentielle uniquement pour la
            # stabilité numérique (évite les overflow avec exp() de grands
            # nombres) ; le résultat mathématique est strictement identique.
            exp_ys = np.exp(ys[t] - np.max(ys[t]))
            ps[t] = exp_ys / np.sum(exp_ys)

        return xs, hs, ys, ps

    def backward(self, xs, hs, ps, targets):
        """
        Backpropagation Through Time (BPTT).

        Idée : on "déroule" le RNN sur les T pas de temps comme si c'était
        un réseau très profond (une couche par pas de temps), PUIS on
        applique la rétropropagation classique, en partant du DERNIER
        pas de temps et en remontant vers le PREMIER (d'où "reversed").

        Un point clé : comme les poids (Wxh, Whh, Why) sont PARTAGÉS entre
        tous les pas de temps, leurs gradients doivent être ACCUMULÉS
        (sommés) sur tous les t, et pas juste calculés une fois.

        Paramètres
        ----------
        xs, hs, ps : dictionnaires retournés par forward()
        targets    : liste de longueur T contenant, pour chaque pas de temps,
                     l'INDICE (entier) de la classe attendue (ex: l'indice
                     du caractère qui aurait dû être prédit).

        Retourne
        --------
        loss  : la perte totale (cross-entropy) sur la séquence
        grads : dict contenant les gradients de chaque paramètre
        hs[T-1] : le dernier état caché (à réutiliser comme h_prev pour
                   le prochain batch, afin de préserver la mémoire)
        """
        # On initialise les gradients à zéro : ils vont être ACCUMULÉS
        # au fur et à mesure qu'on remonte le temps (t = T-1, T-2, ..., 0)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # dh_next représente le gradient qui "vient du futur" : la
        # contribution du pas de temps t+1 au gradient de h_t.
        # Au tout dernier pas de temps (t = T-1), il n'y a pas de futur,
        # donc on l'initialise à zéro.
        dh_next = np.zeros((self.hidden_size, 1))

        loss = 0.0
        T = len(xs)

        # On parcourt le temps À L'ENVERS : c'est le coeur du BPTT.
        # Le gradient à l'instant t dépend du gradient à l'instant t+1
        # (via dh_next), donc on DOIT calculer t+1 avant t.
        for t in reversed(range(T)):

            # --- 1) Perte cross-entropy pour ce pas de temps ---
            # La cross-entropy pénalise le réseau proportionnellement à
            # -log(probabilité qu'il a attribuée à la bonne classe).
            # +1e-12 évite un log(0) si jamais la proba prédite est nulle.
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

            # --- 2) Gradient de la perte par rapport aux logits y_t ---
            # Propriété mathématique très pratique : quand on combine
            # softmax + cross-entropy, la dérivée se simplifie en :
            #     dL/dy = p - one_hot(target)
            # (le "-1" sur la classe correcte vient de cette formule)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            # --- 3) Gradients de la couche de sortie (Why, by) ---
            # y_t = Why @ h_t + by  =>  dWhy = dy @ h_t^T ,  dby = dy
            dWhy += dy @ hs[t].T
            dby += dy

            # --- 4) Gradient qui redescend vers l'état caché h_t ---
            # h_t influence la sortie de DEUX façons :
            #   a) directement, via y_t = Why @ h_t + by
            #   b) indirectement, via h_{t+1} = tanh(... + Whh @ h_t + ...)
            # On additionne donc les deux contributions :
            dh = self.Why.T @ dy + dh_next

            # --- 5) On "passe" à travers la non-linéarité tanh ---
            # Dérivée de tanh(z) par rapport à z : 1 - tanh(z)^2
            # Ici tanh(z) = h_t, donc la dérivée vaut (1 - h_t^2).
            # C'est la règle de la chaîne (chain rule) appliquée à tanh.
            dh_raw = (1 - hs[t] ** 2) * dh

            # --- 6) Gradients des poids qui ont produit h_t ---
            # h_t = tanh(Wxh @ x_t + Whh @ h_{t-1} + bh)
            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T        # contribution entrée -> caché
            dWhh += dh_raw @ hs[t - 1].T    # contribution caché(t-1) -> caché(t)

            # --- 7) Gradient à transmettre au pas de temps PRÉCÉDENT ---
            # C'est ce qui deviendra "dh_next" à l'itération suivante
            # (t-1), représentant l'influence de h_{t-1} sur toute la
            # suite de la séquence.
            dh_next = self.Whh.T @ dh_raw

        # --- Gradient clipping ---
        # Les RNN sont sujets à l'explosion des gradients : comme le même
        # Whh est multiplié à répétition à chaque pas de temps remonté,
        # les gradients peuvent croître exponentiellement sur de longues
        # séquences. On "coupe" (clip) chaque valeur du gradient dans
        # l'intervalle [-5, 5] pour stabiliser l'entraînement.
        for grad in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(grad, -5, 5, out=grad)

        grads = {"Wxh": dWxh, "Whh": dWhh, "Why": dWhy, "bh": dbh, "by": dby}

        # On retourne aussi le dernier état caché : utile pour que
        # l'appel suivant de forward() puisse continuer la mémoire là
        # où elle s'est arrêtée (au lieu de repartir de zéro).
        return loss, grads, hs[T - 1]

    def update_params(self, grads, lr=0.1):
        """
        Met à jour les poids par descente de gradient stochastique (SGD).

        Règle de mise à jour classique : theta = theta - lr * dL/dtheta
        (on avance dans la direction OPPOSÉE au gradient, car le gradient
        pointe vers où la perte AUGMENTE le plus vite).

        lr (learning rate) : taille du pas. Trop grand -> l'entraînement
        diverge. Trop petit -> l'entraînement est très lent.
        """
        self.Wxh -= lr * grads["Wxh"]
        self.Whh -= lr * grads["Whh"]
        self.Why -= lr * grads["Why"]
        self.bh -= lr * grads["bh"]
        self.by -= lr * grads["by"]

    def sample(self, h_prev, seed_idx, n, vocab_size):
        """
        Génère une séquence de n indices, caractère par caractère, en
        utilisant le réseau ENTRAÎNÉ de façon AUTOREGRESSIVE : la sortie
        prédite à un pas de temps devient l'entrée du pas de temps suivant.

        Paramètres
        ----------
        h_prev     : état caché de départ (souvent un vecteur de zéros)
        seed_idx   : indice du premier caractère ("graine" de la génération)
        n          : nombre de caractères à générer
        vocab_size : taille du vocabulaire (pour construire les vecteurs one-hot)

        Retourne
        --------
        indices : liste de n indices représentant les caractères générés
        """
        # Encodage one-hot du caractère de départ
        x = np.zeros((vocab_size, 1))
        x[seed_idx] = 1
        h = h_prev
        indices = []

        for _ in range(n):
            # Un seul pas de forward pass (mêmes équations que dans forward())
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = np.exp(y - np.max(y))
            p = p / np.sum(p)

            # Échantillonnage ALÉATOIRE selon la distribution de probabilité p,
            # plutôt que de prendre systématiquement l'argmax (le plus probable).
            # Cela rend le texte généré plus varié/naturel et évite les
            # répétitions en boucle typiques d'une génération purement gloutonne.
            idx = np.random.choice(range(vocab_size), p=p.ravel())

            # Le caractère généré devient l'entrée du pas de temps suivant
            x = np.zeros((vocab_size, 1))
            x[idx] = 1
            indices.append(idx)

        return indices


# ---------------------------------------------------------------------------
# Exemple d'utilisation : prédiction du caractère suivant dans un texte
# (tâche classique pour tester un RNN "from scratch")
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Préparation des données ---
    # Petit corpus d'exemple. En pratique on utiliserait un texte bien
    # plus long pour obtenir un modèle qui généralise correctement.
    data = "bonjour tout le monde bonjour a tous "

    # Vocabulaire = ensemble des caractères uniques du texte
    chars = sorted(list(set(data)))
    vocab_size = len(chars)

    # Correspondance caractère <-> indice, nécessaire pour l'encodage one-hot
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # --- Hyperparamètres ---
    hidden_size = 64       # taille de la mémoire du RNN
    seq_length = 8         # longueur des sous-séquences utilisées à chaque étape d'entraînement
    learning_rate = 0.1    # pas de la descente de gradient
    n_iterations = 2000     # nombre d'itérations d'entraînement

    rnn = RNN(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size)

    h_prev = np.zeros((hidden_size, 1))  # état caché initial (aucune mémoire au début)
    pointer = 0  # position actuelle dans le texte d'entraînement

    for it in range(n_iterations):
        # Si on atteint la fin du texte, on recommence au début et on
        # réinitialise la mémoire (nouvelle "phrase", pas de continuité logique)
        if pointer + seq_length + 1 >= len(data):
            pointer = 0
            h_prev = np.zeros((hidden_size, 1))

        # On découpe une fenêtre de texte : les entrées sont les seq_length
        # caractères courants, et les cibles sont les MÊMES caractères
        # décalés d'une position vers la droite (prédire le "prochain" caractère)
        input_chars = data[pointer: pointer + seq_length]
        target_chars = data[pointer + 1: pointer + seq_length + 1]

        # Encodage one-hot de chaque caractère d'entrée
        inputs = []
        for ch in input_chars:
            x = np.zeros((vocab_size, 1))
            x[char_to_idx[ch]] = 1
            inputs.append(x)

        # Les cibles restent de simples indices (pas besoin de one-hot,
        # car la cross-entropy n'utilise que l'indice de la bonne classe)
        targets = [char_to_idx[ch] for ch in target_chars]

        # --- Un pas d'entraînement complet ---
        xs, hs, ys, ps = rnn.forward(inputs, h_prev)          # 1) forward pass
        loss, grads, h_prev = rnn.backward(xs, hs, ps, targets)  # 2) backward pass (BPTT)
        rnn.update_params(grads, lr=learning_rate)             # 3) mise à jour des poids

        if it % 200 == 0:
            print(f"Itération {it:4d} | Loss = {loss:.4f}")

        # On avance dans le texte pour le prochain batch
        pointer += seq_length

    # --- Génération de texte après entraînement ---
    # On repart d'un état caché vide et du premier caractère du corpus,
    # puis on laisse le réseau générer la suite lui-même.
    print("\n--- Texte généré ---")
    seed_idx = char_to_idx[data[0]]
    sample_idx = rnn.sample(np.zeros((hidden_size, 1)), seed_idx, n=50, vocab_size=vocab_size)
    generated_text = "".join(idx_to_char[i] for i in sample_idx)
    print(generated_text)
