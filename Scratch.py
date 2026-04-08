
# RÉGRESSION LINÉAIRE - FROM SCRATCH
# (Simple, Multiple, Polynomiale)


import numpy as np
import matplotlib.pyplot as plt


#  1. RÉGRESSION LINÉAIRE SIMPLE


# ALGORITHME :
# 1. Générer dataset (X, y)
# 2. Ajouter biais (colonne de 1)
# 3. Calculer θ = (XᵀX)^(-1) Xᵀy
# 4. Prédire y_pred = Xθ

np.random.seed(0)

X = 10 * np.random.rand(100, 1)
y = 5 + 3*X + np.random.randn(100,1)

# Ajouter biais
X_b = np.c_[np.ones((100,1)), X]

# Calcul des paramètres
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("===== Régression Linéaire Simple =====")
print("b0:", theta[0][0])
print("b1:", theta[1][0])

# Prédiction
y_pred = X_b @ theta

# Visualisation
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Régression Linéaire Simple")
plt.show()

#  2. RÉGRESSION LINÉAIRE MULTIPLE

# ALGORITHME :
# 1. Générer plusieurs variables X1, X2
# 2. Combiner en matrice X
# 3. Ajouter biais
# 4. Calculer θ = (XᵀX)^(-1) Xᵀy
# 5. Prédire


np.random.seed(1)

X1 = np.random.rand(100,1) * 10
X2 = np.random.rand(100,1) * 5

y = 4 + 2*X1 + 3*X2 + np.random.randn(100,1)

# Combiner variables
X_multi = np.hstack((X1, X2))

# Ajouter biais
X_b_multi = np.c_[np.ones((100,1)), X_multi]

# Calcul des paramètres
theta_multi = np.linalg.inv(X_b_multi.T @ X_b_multi) @ X_b_multi.T @ y

print("\n===== Régression Linéaire Multiple =====")
print("b0:", theta_multi[0][0])
print("b1:", theta_multi[1][0])
print("b2:", theta_multi[2][0])

# Prédiction
y_pred_multi = X_b_multi @ theta_multi

# Visualisation (réel vs prédiction)
plt.figure()
plt.scatter(y, y_pred_multi)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.title("Régression Multiple (Réel vs Prédiction)")
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.show()

#  3. RÉGRESSION POLYNOMIALE

# ALGORITHME :
# 1. Générer dataset (X, y)
# 2. Transformer X → (x, x²)
# 3. Ajouter biais
# 4. Calculer θ = (XᵀX)^(-1) Xᵀy
# 5. Prédire

np.random.seed(2)

X = 5 * np.random.rand(100,1)
y = 2 + 1*X + 3*(X**2) + np.random.randn(100,1)

# Transformation polynomiale
X_poly = np.hstack((X, X**2))

# Ajouter biais
X_b_poly = np.c_[np.ones((100,1)), X_poly]

# Calcul des paramètres
theta_poly = np.linalg.inv(X_b_poly.T @ X_b_poly) @ X_b_poly.T @ y

print("\n===== Régression Polynomiale =====")
print("b0:", theta_poly[0][0])
print("b1:", theta_poly[1][0])
print("b2:", theta_poly[2][0])

# Prédiction
y_pred_poly = X_b_poly @ theta_poly

# Visualisation
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
X_range_poly = np.hstack((X_range, X_range**2))
X_range_b = np.c_[np.ones((100,1)), X_range_poly]

y_range = X_range_b @ theta_poly

plt.figure()
plt.scatter(X, y)
plt.plot(X_range, y_range)
plt.title("Régression Polynomiale")
plt.show()


