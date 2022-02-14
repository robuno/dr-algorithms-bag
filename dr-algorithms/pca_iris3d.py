from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

iris = load_iris()
X, y  = iris.data, iris.target

print("Number of dimensions before PCA: "+str(len(X[0])))

pca = PCA(n_components=3)
X_transformed = pca.fit_transform(X)

print("Number of dimensions after PCA: "+str(len(X_transformed[0])))
print("Explained Variance Ratio:",pca.explained_variance_ratio_)



fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
color_map = plt.get_cmap('spring')

scatter_plot =ax.scatter3D(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2],
                           c= X_transformed[:, 0] + X_transformed[:, 1] + X_transformed[:, 2], cmap = color_map)
plt.colorbar(scatter_plot)
plt.title("Iris_DS_3dim_PCA")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
#plt.savefig("Iris_DS_3dim_PCA")
plt.show()
