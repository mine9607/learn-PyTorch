import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from pandas.core.apply import AggObjType
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_samples

X, y = make_blobs(
    n_samples=150,
    n_features=2,
    centers=3,
    cluster_std=0.5,
    shuffle=True,
    random_state=0,
)

plt.scatter(X[:, 0], X[:, 1], c="white", marker="o", edgecolor="black", s=50)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid()
plt.tight_layout()
plt.show()

km = KMeans(
    n_clusters=3, init="random", n_init=10, max_iter=300, tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

plt.scatter(
    X[y_km == 0, 0],
    X[y_km == 0, 1],
    s=50,
    c="lightgreen",
    marker="s",
    edgecolor="black",
    label="Cluster 1",
)
plt.scatter(
    X[y_km == 1, 0],
    X[y_km == 1, 1],
    s=50,
    c="orange",
    marker="o",
    edgecolor="black",
    label="Cluster 2",
)
plt.scatter(
    X[y_km == 2, 0],
    X[y_km == 2, 1],
    s=50,
    c="lightblue",
    marker="v",
    edgecolor="black",
    label="Cluster 3",
)
plt.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    s=250,
    marker="*",
    c="red",
    edgecolor="black",
    label="Centroids",
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()

print(f"Distortion: {km.inertia_:.2f}")

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.tight_layout()
plt.show()


km = KMeans(
    n_clusters=3, init="k-means++", n_init=10, max_iter=300, tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")

y_ax_lower, y_ax_upper = 0, 0
y_ticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(
        range(y_ax_lower, y_ax_upper),
        c_silhouette_vals,
        height=1.0,
        edgecolor="none",
        color=color,
    )
    y_ticks.append((y_ax_lower + y_ax_upper) / 2.0)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(y_ticks, cluster_labels + 1)
plt.xlabel("Silhouette coefficient")
plt.tight_layout()
plt.show()

km = KMeans(
    n_clusters=2, init="k-means++", n_init=10, max_iter=300, tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
plt.scatter(
    X[y_km == 0, 0],
    X[y_km == 0, 1],
    s=50,
    c="lightgreen",
    marker="s",
    edgecolor="black",
    label="Cluster 1",
)
plt.scatter(
    X[y_km == 1, 0],
    X[y_km == 1, 1],
    s=50,
    c="orange",
    marker="o",
    edgecolor="black",
    label="Cluster 2",
)
plt.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    s=250,
    marker="*",
    c="red",
    label="Centroids",
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")

y_ax_lower, y_ax_upper = 0, 0
y_ticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(
        range(y_ax_lower, y_ax_upper),
        c_silhouette_vals,
        height=1.0,
        edgecolor="none",
        color=color,
    )
    y_ticks.append((y_ax_lower + y_ax_upper) / 2.0)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(y_ticks, cluster_labels + 1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette coefficient")
plt.tight_layout()
plt.show()

np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID0", "ID1", "ID2", "ID3", "ID4"]
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

row_dist = pd.DataFrame(
    squareform(pdist(df, metric="euclidean")), columns=labels, index=labels
)

print(row_dist)

row_clusters = linkage(row_dist, method="complete", metric="euclidean")
row_clusters = linkage(df.values, method="complete", metric="euclidean")

df2 = pd.DataFrame(
    row_clusters,
    columns=["row label 1", "row label 2", "distance", "no. of items in clust."],
    index=[f"cluster {(i+1)}" for i in range(row_clusters.shape[0])],
)

print(df2)

# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])

row_dendr = dendrogram(
    row_clusters,
    labels=labels,
    # make dendrogram black (part 2/2)
    # color_threshold=np.inf
)

plt.tight_layout()
plt.ylabel("Euclidean distance")
plt.show()

fig = plt.figure(figsize=(8, 8), facecolor="white")
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation="left")

df_rowclust = df.iloc[row_dendr["leaves"][::-1]]

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation="nearest", cmap="hot_r")

axd.set_xticks([])
axd.set_yticks([])

for i in axd.spines.values():
    i.set_visible(False)

fig.colorbar(cax)
axm.set_xticklabels([""] + list(df_rowclust.columns))
axm.set_xticklabels([""] + list(df_rowclust.index))
plt.show()


ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete")
labels = ac.fit_predict(X)
print(f"Cluster labels: {labels}")

ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")
labels = ac.fit_predict(X)
print(f"Cluster labels: {labels}")

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

plt.scatter(X[:, 0], X[:, 1])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(
    X[y_km == 0, 0],
    X[y_km == 0, 1],
    c="lightblue",
    edgecolor="black",
    marker="o",
    s=40,
    label="cluster 1",
)
ax1.scatter(
    X[y_km == 1, 0],
    X[y_km == 1, 1],
    c="red",
    edgecolor="black",
    marker="s",
    s=40,
    label="cluster 2",
)

ax1.set_title("K-means clustering")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")

ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")

y_ac = ac.fit_predict(X)
ax2.scatter(
    X[y_km == 0, 0],
    X[y_km == 0, 1],
    c="lightblue",
    edgecolor="black",
    marker="o",
    s=40,
    label="cluster 1",
)
ax2.scatter(
    X[y_km == 1, 0],
    X[y_km == 1, 1],
    c="red",
    edgecolor="black",
    marker="s",
    s=40,
    label="cluster 2",
)

ax2.set_title("Agglomerative clustering")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()

db = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
y_db = db.fit_predict(X)
plt.scatter(
    X[y_db == 0, 0],
    X[y_db == 0, 1],
    c="lightblue",
    edgecolor="black",
    marker="o",
    s=40,
    label="cluster 1",
)
plt.scatter(
    X[y_db == 1, 0],
    X[y_db == 1, 1],
    c="red",
    edgecolor="black",
    marker="s",
    s=40,
    label="cluster 2",
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
