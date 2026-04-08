
# MACHINE LEARNING - RÉGRESSIONS (LIBRARY)
# Simple, Multiple, Polynomiale

# 1. IMPORTATION DES LIBRARIES


import numpy as np                     # pour les calculs numériques (tableaux, matrices)
import matplotlib.pyplot as plt       # pour les graphiques
from sklearn.linear_model import LinearRegression   # modèle de régression linéaire
from sklearn.model_selection import train_test_split # pour diviser les données
from sklearn.preprocessing import PolynomialFeatures # pour créer x², x³...
from sklearn.metrics import mean_squared_error, r2_score # pour évaluer le modèle


#  2. RÉGRESSION LINÉAIRE SIMPLE

# ALGORITHME :
# 1. Générer dataset avec 1 variable
# 2. Séparer X et y
# 3. Split train/test
# 4. Entraîner modèle
# 5. Prédire
# 6. Évaluer

np.random.seed(0)   # fixer la graine pour avoir les mêmes résultats à chaque exécution

X = 10 * np.random.rand(100, 1)   # générer 100 valeurs aléatoires entre 0 et 10 (variable X)
y = 5 + 3*X + np.random.randn(100,1)  # y = 5 + 3x + bruit (relation linéaire)

# division des données en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()   # créer le modèle de régression
model.fit(X_train, y_train)  # entraîner le modèle sur les données d'entraînement

y_pred = model.predict(X_test)   # prédire les valeurs sur les données test

# afficher les paramètres
print("\n===== Régression Linéaire Simple =====")
print("Intercept:", model.intercept_)   # b0 (constante)
print("Coefficient:", model.coef_)     # b1 (pente)

# évaluation
print("MSE:", mean_squared_error(y_test, y_pred))  # erreur moyenne
print("R²:", r2_score(y_test, y_pred))            # qualité du modèle

# visualisation
plt.scatter(X_test, y_test)     # afficher les points réels
plt.plot(X_test, y_pred)        # tracer la droite de régression
plt.title("Régression Linéaire Simple")  # titre
plt.show()                      # afficher le graphique


#  3. RÉGRESSION LINÉAIRE MULTIPLE

# ALGORITHME :
# 1. Générer dataset avec plusieurs variables
# 2. Combiner variables en X
# 3. Split train/test
# 4. Entraîner modèle
# 5. Prédire
# 6. Évaluer

np.random.seed(1)   # fixer graine

n = 120   # nombre d'exemples

X1 = np.random.rand(n,1)*10   # variable 1 (ex: heures d'étude)
X2 = np.random.rand(n,1)*5    # variable 2 (ex: sommeil)
X3 = np.random.randint(1,5,(n,1))  # variable 3 (ex: révisions)

y = 10 + 2*X1 + 3*X2 + 4*X3 + np.random.randn(n,1)*2  # relation linéaire multiple

X = np.hstack((X1, X2, X3))   # combiner les variables en une matrice X

# division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()   # créer modèle
model.fit(X_train, y_train)  # entraîner

y_pred = model.predict(X_test)  # prédire

# résultats
print("\n===== Régression Linéaire Multiple =====")
print("Intercept:", model.intercept_)   # constante
print("Coefficients:", model.coef_)     # coefficients des variables

# évaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# visualisation réel vs prédiction
plt.scatter(y_test, y_pred)   # comparer réel et prédiction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])  # ligne parfaite
plt.title("Régression Multiple (Réel vs Prédit)")
plt.show()


#  4. RÉGRESSION POLYNOMIALE

# ALGORITHME :
# 1. Générer dataset non linéaire
# 2. Transformer X avec PolynomialFeatures
# 3. Split train/test
# 4. Entraîner modèle
# 5. Prédire
# 6. Visualiser courbe


np.random.seed(2)   # fixer graine

X = 5 * np.random.rand(100,1)   # variable X
y = 3 + 2*X + 4*(X**2) + np.random.randn(100,1)*3  # relation polynomiale

poly = PolynomialFeatures(degree=2)   # créer transformation polynomiale (x, x²)
X_poly = poly.fit_transform(X)        # transformer X → [1, x, x²]

# division des données
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)

model = LinearRegression()   # créer modèle
model.fit(X_train, y_train)  # entraîner

y_pred = model.predict(X_test)  # prédire

# résultats
print("\n===== Régression Polynomiale =====")
print("Intercept:", model.intercept_)   # constante
print("Coefficients:", model.coef_)     # coefficients

# évaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# visualisation de la courbe
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)  # créer une plage de valeurs
X_range_poly = poly.transform(X_range)   # transformer en polynôme
y_range = model.predict(X_range_poly)    # prédire sur cette plage

plt.scatter(X, y)   # points réels
plt.plot(X_range, y_range)   # courbe
plt.title("Régression Polynomiale")
plt.show()


