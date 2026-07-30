"""
==========================================================
IMPLÉMENTATION DU PERCEPTRON FROM SCRATCH
==========================================================
Le perceptron est l'un des algorithmes de classification 
les plus simples en apprentissage automatique. Il s'agit 
d'un classifieur linéaire binaire, ancêtre des réseaux 
de neurones modernes.

Principe : 
- On combine linéairement les entrées avec des poids (w) 
  et un biais (b) : z = w1*x1 + w2*x2 + ... + b
- On applique une fonction d'activation (ici une fonction 
  en escalier) pour obtenir une sortie binaire (0 ou 1)
- On ajuste les poids itérativement selon l'erreur commise
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """
    Classe implémentant l'algorithme du Perceptron.
    
    Paramètres
    ----------
    learning_rate : float
        Taux d'apprentissage, contrôle l'amplitude de la 
        mise à jour des poids à chaque itération.
        (valeur typique : entre 0.001 et 0.1)
    
    n_iterations : int
        Nombre de fois où l'on parcourt l'ensemble des 
        données d'entraînement (aussi appelé "epochs").
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # Taux d'apprentissage : plus il est grand, plus les 
        # mises à jour des poids seront importantes à chaque étape
        self.learning_rate = learning_rate
        
        # Nombre d'itérations sur l'ensemble du dataset
        self.n_iterations = n_iterations
        
        # Les poids seront initialisés dans la méthode fit()
        # car on a besoin de connaître le nombre de features (colonnes de X)
        self.weights = None
        
        # Le biais permet de décaler la frontière de décision 
        # (équivalent à l'ordonnée à l'origine dans une droite)
        self.bias = None
        
        # Historique des erreurs, utile pour visualiser la convergence
        self.errors_history = []

    def activation(self, x):
        """
        Fonction d'activation en escalier (step function).
        
        Renvoie 1 si x >= 0, sinon 0.
        C'est cette fonction non-linéaire qui transforme 
        la combinaison linéaire en une décision binaire.
        
        Paramètres
        ----------
        x : float ou np.array
            La sortie linéaire (z = w.x + b)
        
        Retour
        ------
        0 ou 1 (ou un tableau de 0 et 1 si x est un vecteur)
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Entraîne le perceptron sur les données fournies.
        
        Paramètres
        ----------
        X : np.array de forme (n_samples, n_features)
            Les données d'entraînement (features/entrées)
        
        y : np.array de forme (n_samples,)
            Les labels/étiquettes correspondants (0 ou 1)
        """
        
        # On récupère le nombre d'exemples et le nombre de features
        n_samples, n_features = X.shape

        # ------------------------------------------------
        # ÉTAPE 1 : Initialisation des paramètres
        # ------------------------------------------------
        # On initialise les poids à zéro (un poids par feature)
        self.weights = np.zeros(n_features)
        
        # Le biais démarre également à zéro
        self.bias = 0

        # ------------------------------------------------
        # ÉTAPE 2 : Boucle d'entraînement
        # ------------------------------------------------
        # On répète l'apprentissage n_iterations fois sur 
        # l'ensemble du dataset
        for iteration in range(self.n_iterations):
            
            # Compteur d'erreurs pour cette itération (epoch)
            errors_this_epoch = 0
            
            # On parcourt chaque exemple du dataset un par un
            # (c'est ce qu'on appelle l'apprentissage "en ligne" 
            # ou "stochastique", par opposition au traitement par lots)
            for idx, x_i in enumerate(X):
                
                # ---- Propagation avant (forward pass) ----
                # Calcul de la combinaison linéaire : z = w.x + b
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Application de la fonction d'activation pour 
                # obtenir la prédiction (0 ou 1)
                y_predicted = self.activation(linear_output)

                # ---- Calcul de l'erreur ----
                # Différence entre la vraie valeur et la prédiction
                # Si prédiction correcte : erreur = 0 (pas de mise à jour)
                # Si erreur : différence = +1 ou -1
                error = y[idx] - y_predicted
                
                # ---- Règle de mise à jour du Perceptron ----
                # Formule : w = w + learning_rate * (y_vrai - y_predit) * x
                #           b = b + learning_rate * (y_vrai - y_predit)
                update = self.learning_rate * error
                
                # Mise à jour des poids : on ajuste chaque poids 
                # proportionnellement à l'entrée correspondante
                self.weights += update * x_i
                
                # Mise à jour du biais (pas de multiplication par x 
                # car le biais n'est associé à aucune feature)
                self.bias += update
                
                # Si une erreur a été commise, on l'ajoute au compteur
                if error != 0:
                    errors_this_epoch += 1
            
            # On stocke le nombre d'erreurs de cette epoch pour 
            # pouvoir visualiser la convergence plus tard
            self.errors_history.append(errors_this_epoch)
            
            # ------------------------------------------------
            # OPTIMISATION : arrêt anticipé si convergence
            # ------------------------------------------------
            # Si aucune erreur n'a été commise sur toute l'epoch, 
            # le modèle a convergé, inutile de continuer
            if errors_this_epoch == 0:
                print(f"Convergence atteinte à l'itération {iteration + 1}")
                break

    def predict(self, X):
        """
        Prédit les classes pour de nouvelles données.
        
        Paramètres
        ----------
        X : np.array de forme (n_samples, n_features)
        
        Retour
        ------
        np.array de forme (n_samples,) contenant les prédictions (0 ou 1)
        """
        # Calcul de la sortie linéaire pour tous les exemples à la fois
        # (grâce au produit matriciel, plus efficace qu'une boucle)
        linear_output = np.dot(X, self.weights) + self.bias
        
        # Application de la fonction d'activation
        return self.activation(linear_output)

    def plot_decision_boundary(self, X, y):
        """
        Trace les points de données et la frontière de décision 
        apprise par le perceptron (fonctionne uniquement pour 
        des données à 2 features, pour pouvoir les visualiser en 2D).
        """
        plt.figure(figsize=(8, 6))
        
        # Affichage des points selon leur classe (0 ou 1)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], 
                    color='red', label='Classe 0', edgecolor='k')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], 
                    color='blue', label='Classe 1', edgecolor='k')
        
        # ------------------------------------------------
        # Calcul de la frontière de décision
        # ------------------------------------------------
        # La frontière est définie par : w1*x1 + w2*x2 + b = 0
        # On isole x2 : x2 = -(w1*x1 + b) / w2
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_values = np.linspace(x1_min, x1_max, 100)
        
        # Protection contre une division par zéro si w2 == 0
        if self.weights[1] != 0:
            x2_values = -(self.weights[0] * x1_values + self.bias) / self.weights[1]
            plt.plot(x1_values, x2_values, 'g--', label='Frontière de décision')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Perceptron : frontière de décision')
        plt.legend()
        plt.grid(True)
        plt.show()


# ==========================================================
# EXEMPLE D'UTILISATION : apprentissage de la porte AND
# ==========================================================
if __name__ == "__main__":
    
    # ------------------------------------------------
    # Données d'entraînement : table de vérité de AND
    # ------------------------------------------------
    # x1 | x2 | AND(x1, x2)
    #  0 |  0 |     0
    #  0 |  1 |     0
    #  1 |  0 |     0
    #  1 |  1 |     1
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    # ------------------------------------------------
    # Création et entraînement du modèle
    # ------------------------------------------------
    perceptron = Perceptron(learning_rate=0.1, n_iterations=20)
    perceptron.fit(X, y)

    # ------------------------------------------------
    # Évaluation du modèle
    # ------------------------------------------------
    predictions = perceptron.predict(X)
    
    print("\n--- Résultats ---")
    print("Prédictions   :", predictions)
    print("Vraies valeurs:", y)
    print("Poids finaux  :", perceptron.weights)
    print("Biais final   :", perceptron.bias)
    
    # Calcul de la précision (accuracy)
    accuracy = np.mean(predictions == y)
    print(f"Précision     : {accuracy * 100:.2f}%")

    # ------------------------------------------------
    # Visualisation de la frontière de décision
    # ------------------------------------------------
    perceptron.plot_decision_boundary(X, y)
