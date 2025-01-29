import numpy as np
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

    def __init__(self, K,data ,max_iters):
        self.data = data # 2D array (m,n)
        self.K = K # number of clusters
        self.max_iters = max_iters # max iterrations for moving the centroids
        self.labled_points = np.zeros(data.shape[0]) # 1D array of size m, each index corresponds to point with same index in data.
        self.centroids = [] # list of points(1D Arrays of size n) which are the centers of each cluster.
        
        
    def predict(self):
        '''
        Description:
            Assiging cluster for each point in the given data.
            The assignment is provided by returning a labels array - index of label i corresponding to vector i in data.


        Return value:
            self.labled_points - 1D array of size m, each index corresponds to point with same index in data.
            
        Example:
            self.data: [[1,2,3], [4,5,6], [7,8,9]]
            self.K : 2
            Return labels:  [0,0,1] (v1,v2 assigend to cluster 0 and v3 assigned to cluster 1)
        '''
        
        # Init centroids with Kmeans++ algorithm  
        #   self.centroids = list of points (1D arrays of size n) which will be promoted to centroids.
        self.centroids = calculate_initial_centroids(self.data, self.K)
   
        # Update centroids based on kmeans algorithm
        for _ in range(self.max_iters):
            
            """Assign Cluster To Points"""
            # Cluster the labels based on given centorids
            #   self.labled_points - # 1D array of size m, each index corresponds to point with same index in data.
            self.labled_points = self._create_clusters(self.centroids)
            
            
            """Calculate New Centroid Points"""
            # Calculate new centroids from the clusters
            centroids_old = self.centroids # Save old centroid locations for comparison
            
            # calculate the new value for centroids
            #   self.centroids - list of points(1D Arrays of size n) which are the centers of each cluster.
            self.centroids = self._get_centroids(self.labled_points) 
            
            """Test If Done"""
            # Check if centroids moved
            #   by by comapring distnaces between old and new centroid points
            if self._is_converged(centroids_old, self.centroids): 
                break

        return self.labled_points
 

     
    def _create_clusters(self, centroids):
        '''
        Description:
            Assign each point its cluster by calcualting all the distances between all points and all clusters.
            The distacne calculations are preformed elementwise and the results (distacnes) are kept in (m,k,n) array during the process.
            The minimal distance along the centroids axis is choosen for each point, and stored in labled_points.
        
        Parameters:
            centroids - list of points(1D Arrays of size n) which are the centers of each cluster.
            
        Return value:
            labled_points - 1D array of size m, each index corresponds to point with same index in data.
        '''
        
        # Convert list to to NumPy array
        centroids_np_arr = np.array(centroids) 
        
        """Make The Points And Centroids Arrays Compatible"""
        # Important: since centroids are (k,n) and points are (m,n) we need to make theses array compatible for vectorization.
        broadcasted_centroid_array = centroids_np_arr[np.newaxis, :, :] # make the centroids (1, k, n)
        broadcasted_points_array = self.data[:, np.newaxis, :]   # make the points    (m, 1, n)
        # Importent: now centroids and points are compatible for operations (like subtraction, summation and power - to calculate distacne between all points to all centroids)
        
        """Distance Calcualtin"""
        points_minus_centroids = broadcasted_points_array - broadcasted_centroid_array # np.array of shape (m,k,n) of all subtractions
        points_minus_centroids_squared = points_minus_centroids ** 2 # np.array of shape (m,k,n) 
        
        # shape of distance after sum is now (m, k) each cell represants distance of point i(of m) and centroid j (of k)
        distances = np.sum(points_minus_centroids_squared, axis=2)  # axis 2 is features, we want to sum the features to get all dsitances between points and centroids
        
        """Label The Points"""
        # For each point keep the index of cluster with minimal distance
        labled_points = np.argmin(distances, axis=1)  # argmin returns array of m indexes , each index is of the nearest centroid to that point. axis 1 means along the columns (the centroids)
        

        return labled_points



    def _get_centroids(self, labled_points):
        # Number of clusters
        K = self.K
        
        # Initialize a new centroids array (K, n)
        new_centroids = np.zeros((K, self.data.shape[1]))  # Shape (K, n)
    
        # For each cluster (from 0 to K-1)
        for k in range(K):
            # Find all points assigned to the current cluster
            # This is slicing of data by providing the indexing we want to keep:
            #   - only the indexes of points which in labled_points has value k
            cluster_points = self.data[labled_points == k]  # cluster_points is (num_of_points_in_cluster_k, n )
            
            # Calculate the mean of the points in this cluster
            # np.mean returns single point (array of size n) after summing all features of all points in cluster K and deviding by the count
            new_centroids[k] = np.mean(cluster_points, axis=0)  # Mean across the features (axis 0)
    
        return new_centroids
    
    

    # Check if centroids didnt move
    def _is_converged(self, centroids_old, centroids):
        '''
        Description:
            Check if centroids did not moved.

        Parameters:
            centroids_old - list of points representing the old centroids
            centroids     - list of points representing the new centroids
            
        Return value:
            boolean - true if no changes where executed, otherwise false.
            
        Example:
            centroids_old: [[1,2], [4,5], [7,9]]
            centroids:     [[1,2], [4,5], [9,8]] 
            returns flase            
        '''
        # distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)] # find distnace between old and new centroids
        return sum(distances) == 0 # if all centroids didnt move - convergence

  
def calculate_initial_centroids(data, k):
    '''
    Parameters:
        data - # 2D array (m,n)
        k - # number of clusters
    Return value:
        centroids - list of points (1D arrays of size n) which will be promoted to centroids.
    '''
    centroids = []  # Initialize the centroids list
    random_index_of_vector = np.random.randint(data.shape[0]) # Choose index of a point from data randomly
    random_centroid_from_data = data[random_index_of_vector, :] # Pick the vector (with all features) based on the random index
    centroids.append(random_centroid_from_data)  # Add the random centroid
    
    
    ''' Compute locations for remaining centroids '''
    for c_id in range(k - 1):  # Compute remaining k - 1 centroids (1 already chocen randomly)
        dist = []  # Initialize a list to store distances of data points from nearest centroid
        
        ''' Save distance for nearest centroid for each vector '''
        for i in range(data.shape[0]):  # For each vector/point in the dataset we will save the distacne to the nearest centroid and keep it in dist[].
            point = data[i, :]  # Point representing vector i by all columns of row i in the dataset
            d = sys.maxsize  # Depending on the system, largest size of array/string

            # Compute distance of current point from each of the previously selected centroid and store the minimum distance
            for j in range(len(centroids)):
                temp_dist = euclidean_distance(point, centroids[j]) # Find distance between point and current centroid
                d = min(d, temp_dist) # update minimum if new centroid is closer
            dist.append(d)
        
        ''' Find point with maximal distance from its centroid '''
        # Select data point with maximum distance as our next centroid
        dist = np.array(dist)  # Create new vector based on list of minimum distances of all points
        row_index_of_the_point_with_highest_distance = np.argmax(dist) # Get index of point with maximal distance
        next_centroid = data[row_index_of_the_point_with_highest_distance, :]  # Point with maximum distance
        centroids.append(next_centroid)

    return centroids

# Testing
if __name__ == "__main__":
    
    
    # Measure the time taken for the entire process
    start_time = time.time()
    
    # path to full .csv given by Dan
    path_to_csv = r"C:\Users\Oren Baranovsky\Documents\braude\Semester 7\ProgrammingLangauges\FinalProject\input_data_set\kmeans_data.csv"
    data = np.loadtxt(path_to_csv, delimiter=",")  # Assuming the file uses commas as the delimiter
        
    # make sure data is 10K on 100 (if Dans dataset)
    print("shape", data.shape)
    
    distances = pairwise_distances(data)
    
    # try 10 values for k
    for k in range (2,11):
        kmeans = KMeans(K=k,data = data, max_iters=150)
        labels = kmeans.predict()
        #print("labels", type(labels))
        silhouette_avg = compute_silhouette(labels, distances)
        print(f"Silhouette for round {k} is {silhouette_avg}")
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.6f} seconds")
    