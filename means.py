import numpy as np
import time  # Used to see how much time the run takes.
import random # used to randomize float between 0  and 1

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
    '''
    Compute the Silhouette Coefficient.
        labels - array where each entry is the cluster label of the corresponding data point
        distances - precomputed pairwise distance matrix of shape (N_samples, N_samples)
    Return value:
        Silhouette Coefficient
    '''
    # Unique cluster labels
    unique_labels = np.unique(labels) # Returns the unique elements of an array
    
    # Preallocate arrays for cohesion (a) and separation (b)
    N_samples = distances.shape[0] # Returns the number of samples in the dataset
    a = np.zeros(N_samples)
    b = np.full(N_samples, np.inf)  # Initialize with large values for separation
    
    # Dictionary to store indices for each cluster
    cluster_indices = {}
    for label in unique_labels:
        cluster_indices[label] = np.where(labels == label)[0] # Returns the indices where the condition is True 
    # example: cluster_indices = { 0: [1,3,4],  1: [0,2]}
    
    # Precompute cohesion (a) and separation (b)
    for label, indices in cluster_indices.items():
        # example:label - 0, indices - [1,3,4]
        
        # Cohesion (a)
        cluster_distances = distances[np.ix_(indices, indices)] # Extracts a submatrix from distances, selecting rows and columns corresponding to indices
        np.fill_diagonal(cluster_distances, np.nan)  # Exclude self-distances
        a[indices] = np.nanmean(cluster_distances, axis=1)
        
        # Separation (b)
        for other_label, other_indices in cluster_indices.items():
            if label == other_label:
                continue
            # Compute mean distance to other clusters
            
            # Extracts a submatrix from distances, selecting rows corresponding to indices and columns to other indices
            inter_distances = np.mean(distances[np.ix_(indices, other_indices)], axis=1) 
            b[indices] = np.minimum(b[indices], inter_distances)  # Minimum across other clusters
    
    # Compute silhouette scores
    silhouette_scores = (b - a) / np.maximum(a, b)
    return np.nanmean(silhouette_scores)


class KMeans:
    
    
    def __init__(self, K,data):
        self.data = data # 2D array (m,n)
        self.K = K # number of clusters
        self.labled_points = np.zeros(data.shape[0]) # 1D array of size m, each index corresponds to point with same index in data.
        self.centroids = [] # will be a list of points(1D Arrays of size n) which are the centers of each cluster.
        
        
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
        
        '''Init centroids with Kmeans++ algorithm'''  
        self.calculate_initial_centroids() # self.centroids list is now updated with initial points
   
        '''Label The Points With Centroid Indexes'''
        while (True):
            
            """Assign Cluster To Points"""
            # Cluster the labels based on self.centroids
            self.labled_points = self._create_clusters() # self.labled_points is now updated with cluster indexes
            
            """Calculate New Centroid Points"""
            centroids_old = self.centroids # Save old centroid locations for comparison
            
            # calculate the new value for centroids
            self.centroids = self._get_centroids() # shape - (k, n) - 2D array of k centroid points
            
            """Test If Done"""
            # Check if centroids moved by comapring distnaces between old and new centroid points
            if self._is_converged(centroids_old): 
                break

        return self.labled_points
 

     
    def _create_clusters(self):
        '''
        Description:
            Assign each point its cluster by calcualting all the distances between all points and all clusters.
            The distacne calculations are preformed elementwise and the results (distacnes) are kept in (m,k,n) array during the process.
            The minimal distance along the centroids axis is choosen for each point, and stored in labled_points.
              
        Return value:
            labled_points - 1D array of size m, each index corresponds to point with same index in data.
        '''
        
        # Convert list to to NumPy array
        centroids_np_arr = np.array(self.centroids) 
        
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
        
        """Label"""
        # For each point keep the index of cluster with minimal distance
        labled_points = np.argmin(distances, axis=1)  # argmin returns array of m indexes , each index is of the nearest centroid to that point. axis 1 means along the columns (the centroids)
        

        return labled_points
    
        
    def weighted_random_choice(self, minimal_squared_distances):
        '''
        Choose one new data point at random as a new center,
        using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
        '''
        # Get sum of all distqances to be in denominator of probabilities
        squared_distance_sum = np.sum(minimal_squared_distances, axis = 0) # shape - () - Scalar
        
        # Get probability based on distance only
        probabilities = minimal_squared_distances / squared_distance_sum # shape - (m, ) - 1D array
        
        # Update probability to be comulative (sum until the point)
        cumulative_probabilities = np.cumsum(probabilities) # shape - (m, ) - 1D array
        
        # Get random float between 0 and 1
        random_float = random.random()
        
        # Choose point from points - points with high ditacne has proportional chance of picling
        for i, cp in enumerate(cumulative_probabilities):
            if random_float <= cp:
                return i
            
     
    def calculate_initial_centroids(self):
        '''
        Description:
           Kmeans++ algorithm for good initialization of centroids:
           Choose one center uniformly at random among the data points.
           For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
           Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
           Repeat Steps 2 and 3 until k centers have been chosen.    
        '''

        '''Choose one center uniformly at random among the data points.'''
        random_index_of_vector = np.random.randint(self.data.shape[0]) # Choose index of a point from data randomly
        random_centroid_from_data = self.data[random_index_of_vector, :] # Pick the vector (with all features) based on the random index
        self.centroids.append(random_centroid_from_data)  # Add the random centroid
        
        ''' Compute locations for remaining centroids '''
        for _ in range(self.K - 1):  # Compute remaining k - 1 centroids
        
            '''Make Data Compatible For Vectorization'''
            centroids_np_arr = np.array(self.centroids)  # Convert list to numpy array
            data_compatible = data[:, np.newaxis, :]  # Shape: (m, 1, n) - 3D array
            centroids_compatible = centroids_np_arr[np.newaxis, :, :]  # Shape: (1, k, n) - 3D array
            
            '''Compute Squared Distances For All Points'''
            distances_pre_compute = (data_compatible - centroids_compatible) ** 2
            squared_distances = np.sum(distances_pre_compute, axis=2)  # Shape: (m, k) - 2D array
            # Compute the minimum distance to any centroid
            minimal_squared_distances = np.min(squared_distances, axis=1)  # Shape: (m,) - 1D array

            '''Choose Next Centroid Using Weighted Probability'''
            index_of_next_centroid = self.weighted_random_choice(minimal_squared_distances)
            
            '''Add The Centroid'''
            self.centroids.append(self.data[index_of_next_centroid])  # Append new centroid


    def _get_centroids(self):
    
        '''
        Description:
            Calculates avarages of each cluster and return the point located in the avarage
        
        Return:
            new_centroids - 2D array of shape (k, n) of points which are centroids
            
        Example:
            labaled_points: [0,0,0,1,1,2,2]
            k = 3
            n = 2
            retuns new_centroids: [[1,2], [0,2], [3,4]]
        '''  
        
        # Initialize a new centroids array (K, n)
        new_centroids = np.zeros((self.K, self.data.shape[1]))  # Shape (K, n)
    
        # For each cluster (from 0 to K-1)
        for k in range(self.K):
            '''Get only points assigned to the current cluster index''' 
            # This is slicing of data by providing the indexing we want to keep:
            #   - only the indexes of points which in labled_points has value k
            cluster_points = self.data[self.labled_points == k]  # shape - (num_of_points_in_cluster_k, n)
            
            # Calculate the mean of the points in this cluster
            # np.mean returns single point (array of size n) after summing all features of all points in cluster K and deviding by the count
            new_centroids[k] = np.mean(cluster_points, axis=0)  # shape of element in list - (n,)
    
        return new_centroids # shape - (k, n)
    
    
    def _is_converged(self, centroids_old):
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
        distances = [euclidean_distance(centroids_old[i], self.centroids[i]) for i in range(self.K)] # find distnace between old and new centroids
        return sum(distances) == 0 # if all centroids didnt move - convergence

  


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
        kmeans = KMeans(K=k,data = data)
        labels = kmeans.predict()
        #print("labels", type(labels))
        silhouette_avg = compute_silhouette(labels, distances)
        print(f"Silhouette for round {k} is {silhouette_avg}")
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.6f} seconds")
    

    