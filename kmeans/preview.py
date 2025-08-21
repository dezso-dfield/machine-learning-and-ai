import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----- 1) Generate some synthetic data
np.random.seed(42)
group1 = np.random.randn(50, 2) + np.array([2, 2])
group2 = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack([group1, group2])

# ----- 2) Initialize centroids randomly
k = 2
rng = np.random.default_rng(0)
centroids = X[rng.choice(len(X), size=k, replace=False)]

# ----- 3) Helper functions
def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

# ----- 4) Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
colors = ["#4A90E2", "#E24A4A"]

scatter = ax.scatter(X[:, 0], X[:, 1], c="gray", s=40, alpha=0.6)
centroid_scatter = ax.scatter(centroids[:, 0], centroids[:, 1],
                              c=colors, marker="X", s=200, edgecolor="black")

ax.set_title("K-Means Clustering (2 clusters)")
ax.set_xlim(X[:,0].min()-1, X[:,0].max()+1)
ax.set_ylim(X[:,1].min()-1, X[:,1].max()+1)

# Lines to track centroid movement
lines = [ax.plot([], [], c=colors[i], lw=1.5, linestyle="--")[0] for i in range(k)]
paths = [[] for _ in range(k)]

def init():
    scatter.set_array(np.array([]))
    return scatter, centroid_scatter, *lines

def update(frame):
    global centroids
    labels = assign_clusters(X, centroids)
    scatter.set_array(labels)

    new_centroids = update_centroids(X, labels, k)

    # update centroid positions and paths
    for i in range(k):
        paths[i].append(centroids[i])
        path = np.array(paths[i])
        lines[i].set_data(path[:,0], path[:,1])

    centroids[:] = new_centroids
    centroid_scatter.set_offsets(centroids)
    centroid_scatter.set_color(colors)

    return scatter, centroid_scatter, *lines

anim = FuncAnimation(fig, update, frames=6, init_func=init,
                     interval=1000, blit=False, repeat=False)

plt.show()
