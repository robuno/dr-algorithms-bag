from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X, y  = iris.data, iris.target

print("Number of dimensions before PCA: "+str(len(X[0])))

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

print("Number of dimensions after PCA: "+str(len(X_transformed[0])))
print("Explained Variance Ratio:",pca.explained_variance_ratio_)

plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Iris Dataset")
plt.savefig("Iris_DS_2dim_PCA")
plt.show()
