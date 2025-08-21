import numpy as np

# ------------------------------
# K-Means Functions
# ------------------------------
def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iter=100):
    # Randomly pick centroids from data
    rng = np.random.default_rng(0)
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    
    for step in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        
        print(f"\n--- Step {step+1} ---")
        for i in range(k):
            print(f"Cluster {i+1}: Centroid {new_centroids[i]}, "
                  f"Points: {X[labels==i].tolist()}")
        
        # stop if converged
        if np.allclose(centroids, new_centroids, atol=1e-6):
            print("\n✅ Converged!")
            break
        centroids = new_centroids
    return labels, centroids

# ------------------------------
# User Input Section
# ------------------------------
print("K-Means Clustering (User Input)")
print("Enter your points like: x1,y1 x2,y2 x3,y3 ...")
points_str = input("Points: ")
k = int(input("Number of clusters (k): "))

# Parse input into array
points = []
for p in points_str.split():
    x, y = map(float, p.split(","))
    points.append([x, y])
X = np.array(points)

# Run K-Means
labels, centroids = kmeans(X, k)

print("\nFinal Centroids:")
print(centroids)
print("Cluster Assignments:", labels)
