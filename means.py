import numpy as np
import matplotlib.pyplot as plt
import sys   # used for sys.maxsize :Depending on the system, largest size of array/string
import time  # Used to see how much time the run takes.

# In kmeans we can omit the sqrt, since it is monotonic - order of distances kept 
def euclidean_distance(x1, x2):
    return np.sum((x1-x2)**2)


# Computing squared distances between all vectors (rows of the data matrix)
def pairwise_distances(data):
    """
    Returns matrix (m,m) of squared distances between all the vectors
    cell[i][j] in the matrix: ||Xi|| + ||Xj|| - 2 * <Xi,Xj> 
    """
    m, n = data.shape  # m number of rows (points\vectors) and n is the number of columns (features\dimentios of the vectors)
    B = data @ data.T  # Compute dot product of every 2 vectors - <Xi,Xj>, resulting with (m,m) dot products matrix 
    D = np.sum(data ** 2, axis=1) # calcualte the norm of each vector ||Xi||^2 , resulting in 1D array of m vectors.
    D = D.reshape(1, m) # Resahpe for python - D was (m,) which is a 1D array with m elements, we need it to be in form of 2D array (matrix) to calculate elementwise.
    return (D.T + D - 2 * B)  # Calculate (|Xi|^2 - 2 * <xi,xj> +|Xj|^2 ) = (Xi - Xj)^2 for all combinations of vectors and return in (m,m) matrix if squared distnaces.

def compute_silhouette(labels, distances):
    """
    Optimized computation of the Silhouette Coefficient.
    """
    # Unique cluster labels
    unique_labels = np.unique(labels)
    
    # Preallocate arrays for cohesion (a) and separation (b)
    N_samples = distances.shape[0]
    a = np.zeros(N_samples)
    b = np.full(N_samples, np.inf)  # Initialize with large values for separation
    
    # Dictionary to store mask indices for each cluster
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}
    
    # Precompute cohesion (a) and separation (b)
    for label, indices in cluster_indices.items():
        # Cohesion (a)
        cluster_distances = distances[np.ix_(indices, indices)]
        np.fill_diagonal(cluster_distances, np.nan)  # Exclude self-distances
        a[indices] = np.nanmean(cluster_distances, axis=1)
        
        # Separation (b)
        for other_label, other_indices in cluster_indices.items():
            if label == other_label:
                continue
            # Compute mean distance to other clusters
            inter_distances = np.mean(distances[np.ix_(indices, other_indices)], axis=1)
            b[indices] = np.minimum(b[indices], inter_distances)  # Minimum across other clusters
    
    # Compute silhouette scores
    silhouette_scores = (b - a) / np.maximum(a, b)
    return np.nanmean(silhouette_scores)

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K # number of clusters
        self.max_iters = max_iters # max iterrations for moving the centroids 
        self.plot_steps = plot_steps # DEBUG - plot each centroid change

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)] # initiate list of k empty lists. each empty list contians the row indices which represent the vectors in that cluster.

        # the centers (mean vector) for each cluster
        self.centroids = []

    # Main process - devide into clusters and returns the labels based on the rows in the given data
    def predict(self, data):
        self.data = data
        self.n_samples, self.n_features = data.shape # Save number of  (m) and number of features for each vector (n)

        # initialize - instead of random init with chosen centroids
        #random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        #self.centroids = [self.data[idx] for idx in random_sample_idxs]
        
        self.centroids = calculate_initial_centroids(data, self.K)
   

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps: # DEBUG: plot 
                self.plot()

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters) # calculate the new value for centroids

            if self._is_converged(centroids_old, self.centroids): # check if centroids moved
                break

            if self.plot_steps: # Debug - plot
                self.plot()

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    
    # Returns np.array of size m (number of vectors), each vector in its location according to the original data, and the data is the index of the cluster
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples) # Return a new array of given shape and type, without initializing entries.
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    # Creates a list of lists: each list represents a cluster. adds indices of the vectors to the lists based on their distance
    def _create_clusters(self, centroids):
        # Calculate the squared Euclidean distance between each point and each centroid
        distances = np.sum((self.data[:, np.newaxis, :] - centroids) ** 2, axis=2)  # Shape: (m, K)
        cluster_labels = np.argmin(distances, axis=1)  # Find the index of the closest centroid for each point
        clusters = [np.where(cluster_labels == k)[0] for k in range(self.K)]  # Group indices by cluster
        return clusters

    # returns index of the closest centroid in centroids list to the given sample
    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids] # calculte distances from sample to all centroids
        closest_idx = np.argmin(distances) # get the minimum distance from sample to centroid.
        return closest_idx

    # Calculate new value for centroids
    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features)) # create 2D array of 0's, each row represents centroid, each column represents feature.
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.data[cluster], axis=0) # calculate the mean of vectors in cluster element-wise (mean of feature1, feature2....mean of feature 100)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    # Check if centroids didnt move
    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)] # find distnace between old and new centroids
        return sum(distances) == 0 # if all centroids didnt move - convergence

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.data[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


    
    
# Returns a list of k centroids (vectors of length n)
def calculate_initial_centroids(data, k):
    '''
    Parameters:
        data - matrix where each row represents a vector (point in Euclidean space)
        k - number of clusters
    Return value:
        centroids - list of points which will be promoted to centroids.
    '''
    centroids = []  # Initialize the centroids list
    centroids.append(data[np.random.randint(data.shape[0]), :])  # First centroid randomly chosen from data

    for c_id in range(k - 1):  # Compute remaining k - 1 centroids
        dist = []  # Initialize a list to store distances of data points from nearest centroid
        for i in range(data.shape[0]):  # For each vector/point in the dataset
            point = data[i, :]  # Point representing vector i by all columns of row i in the dataset
            d = sys.maxsize  # Depending on the system, largest size of array/string

            # Compute distance of current point from each of the previously selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = euclidean_distance(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)

        # Select data point with maximum distance as our next centroid
        dist = np.array(dist)  # Create new vector based on list of minimum distances of all points
        row_index_of_the_point_with_highest_distance = np.argmax(dist)
        next_centroid = data[row_index_of_the_point_with_highest_distance, :]  # Point with maximum distance
        centroids.append(next_centroid)

    return centroids
# Testing
if __name__ == "__main__":
    
    want_to_plot = False # Change to true only when number of features = 2
    want_random_data = False
    
    # Measure the time taken for the entire process
    start_time = time.time()
    

    if want_to_plot:
        # Path to version with 2 dims only.
        path_to_csv = r"C:\Users\Oren Baranovsky\Documents\braude\Semester 7\ProgrammingLangauges\FinalProject\input_data_set\MyExperiment\given_dataset_but_only_2_dim_decimal.csv"
        with open(path_to_csv, 'r', encoding='utf-8-sig') as file:
            data = np.loadtxt(file, delimiter=",")  # Assuming commas as delimiters
    else:
        # path to full .csv given by Dan
        path_to_csv = r"C:\Users\Oren Baranovsky\Documents\braude\Semester 7\ProgrammingLangauges\FinalProject\input_data_set\kmeans_data.csv"
        data = np.loadtxt(path_to_csv, delimiter=",")  # Assuming the file uses commas as the delimiter
        
    if want_random_data:
        np.random.seed(42)
        from sklearn.datasets import make_blobs
        
        data, y = make_blobs(
            centers=3, n_samples=10000, n_features=2, shuffle=True, random_state=40
        )
        
        
        clusters = len(np.unique(y))
        print(clusters)
        
    # make sure data is 10K on 100 (if Dans dataset)
    print("shape", data.shape)
    
    distances = pairwise_distances(data)
    print(distances)
    
    # try 10 values for k
    for k in range (2,11):
        kmeans = KMeans(K=k, max_iters=150, plot_steps=want_to_plot)
        labels = kmeans.predict(data)
        print("labels", type(labels))
        silhouette_avg = compute_silhouette(labels, distances)
        print(f"Silhouette for round {k} is {silhouette_avg}")
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.6f} seconds")
    