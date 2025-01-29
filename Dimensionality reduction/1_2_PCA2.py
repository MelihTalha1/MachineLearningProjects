from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components = 3) # 3 principal components (PC)
X_pca = pca.fit_transform(X)

fig = plt.figure(1, figsize=(8,6))
ax = fig.add_subplot(111, projection = "3d", elev=-150, azim=110)

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c = y, s = 40)

ax.set_title("First Three PCA Dimesions of Iris Dataset")
ax.set_xlabel("1st Eigenvector")
ax.set_ylabel("2st Eigenvector")
ax.set_zlabel("3st Eigenvector")

plt.show() 