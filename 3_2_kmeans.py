#####
import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions

# %matplotlib inline

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["target"] = iris.target
print(df.head())

#####
iris_df1 = pd.DataFrame(iris.data[:50])
iris_df2 = pd.DataFrame(iris.data[50:100])
iris_df3 = pd.DataFrame(iris.data[100:150])

plt.scatter(iris_df1[0], iris_df1[1], c="red")
plt.scatter(iris_df2[0], iris_df2[1], c="blue")
plt.scatter(iris_df3[0], iris_df3[1], c="green")

plt.title("sepal")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.grid(True)
# plt.show()
plt.savefig("img/sepal_scatter.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Graph saved as img/sepal_scatter.png")

#####
plt.scatter(iris_df1[2], iris_df1[3], c="red")
plt.scatter(iris_df2[2], iris_df2[3], c="blue")
plt.scatter(iris_df3[2], iris_df3[3], c="green")

plt.title("petal length")
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.grid(True)
# plt.show()
plt.savefig("img/petal_scatter.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Graph saved as img/petal_scatter.png")


#####
X = iris.data
# X = iris.data[:, [0, 1]]

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

print(kmeans.labels_)

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=kmeans.labels_)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
# plt.show()
plt.savefig("img/kmeans_sepal.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Graph saved as img/kmeans_sepal.png")

plt.scatter(df.iloc[:, 2], df.iloc[:, 3], c=kmeans.labels_)
plt.xlabel(df.columns[2])
plt.ylabel(df.columns[3])
# plt.show()
plt.savefig("img/kmeans_petal.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Graph saved as img/kmeans_petal.png")
