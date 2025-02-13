# define	NUMBER_OF_CLUSTERS	10
# define	MAXIMUM_ITERATIONS	100
#define		FEATURE_DIM			100
#define		MAX_LINE_LENGTH		10000
#define		_CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>


/*----------------------------------------------------------------------
* Point in FEATURE_DIM dimention euclidian space - vector with 100 features
----------------------------------------------------------------------*/
typedef struct {
	double features[FEATURE_DIM];
	int group;
} POINT;

/*----------------------------------------------------------------------
* Functions definition
----------------------------------------------------------------------*/
POINT* read_points_from_csv(const char* filename, int* num_pts);
double euc_dist_squared(POINT* a, POINT* b);
double** pairwise_distances(POINT* points, int num_pts);
int get_index_of_nearest_centroid(POINT* pt, POINT* centroid_arr, int n_cluster);
void free_distance_matrix(double** distance_matrix, int num_pts);
void init_clusters_with_kpp(POINT* pts, int num_pts, POINT* centroids, int num_clusters);
double get_distance_to_nearest_centroid(POINT* pt, POINT* centroid_arr, int n_cluster);
POINT* kmeans_plus_plus(POINT* pts, int num_pts, int num_clusters, int maxTimes);
double calculate_silhouette_for_point(POINT* pts, int num_pts, int point_index, int k, double** distance_matrix);
double calculate_silhouette_score(POINT* pts, int num_pts, int k, double** distance_matrix);
void print_points_and_centroids(POINT* pts, int num_pts, POINT* centroids, int num_clusters);



/*------------------------------------------------------
* Utility function for reading the CSV data file
  Input:
	const char* filename: path to .csv file
	int*		 num_pts: second return value, integer used to update the caller about the number of points read from the .csv file
  Output:
	POINT*		pts: array of all the points in the data
------------------------------------------------------*/
POINT* read_points_from_csv(const char* filename, int* num_pts) {
	// Open file
	FILE* file = fopen(filename, "r");
	if (!file) {
		perror("Error opening file\n");
		return NULL;
	}

	// Count lines
	*num_pts = 0;
	int ch;
	while ((ch = fgetc(file)) != EOF) {
		if (ch == '\n') (*num_pts)++;
	}
	rewind(file);

	// Allocate memory for points
	POINT* pts = (POINT*)malloc(sizeof(POINT) * (*num_pts));
	if (!pts) {
		printf("Memory allocation failed\n");
		fclose(file);
		return NULL;
	}

	// Read points
	for (int i = 0; i < *num_pts; i++) {
		for (int j = 0; j < FEATURE_DIM; j++) {
			double value;
			if (fscanf(file, "%lf,", &value) != 1) {
				printf("Error reading file\n");
				free(pts);
				fclose(file);
				return NULL;
			}
			pts[i].features[j] = value;
		}
		// Skip potential newline
		fscanf(file, "\n");
	}

	fclose(file);
	return pts;
}

/*----------------------------------------------------------------------
* Calculating euclidain distance between two vectors of dimention 100.
  Input:
		POINT* a, POINT* b: Points to mesure squared euclidian
  Output:
		double distance: squared distance between the two vectors
----------------------------------------------------------------------*/
double euc_dist_squared(POINT* a, POINT* b)
{
	double distance = 0.0;
	for (int i = 0; i < FEATURE_DIM; i++) {
		double diff = a->features[i] - b->features[i];
		distance += diff * diff;
	}
	return distance;
}

/*----------------------------------------------------------------------
* Calculating pairwise distances between all the points in the data.
  Input:
		POINT* points: array of all the points in the data
		int		num_pts: number of points in the data
  Output:
		double** distance_matrix: a matrix where each sell [i,j] holds the Euclidean distance between i and j
----------------------------------------------------------------------*/
double** pairwise_distances(POINT* points, int num_pts) {

	// Allocate memory for the distance matrix
	double** dist_matrix = (double**)malloc(num_pts * sizeof(double*)); //1D
	if (!dist_matrix) {
		printf("Allocation memory failed\n");
		return NULL;
	}
	for (int i = 0; i < num_pts; i++) {
		dist_matrix[i] = (double*)calloc(num_pts, sizeof(double)); //2D

		// Case if memory allocation failed
		if (!dist_matrix[i]) {
			printf("Allocation memory failed\n");
			free_distance_matrix(dist_matrix, num_pts);
			return NULL;
		}
	}

	// Precompute squared norms of each POINT
	double* row_norms = (double*)malloc(num_pts * sizeof(double));
	if (!row_norms) {
		printf("Memory allocation failed");
		free_distance_matrix(dist_matrix, num_pts);
		return NULL;
	}

	for (int i = 0; i < num_pts; i++) {
		double norm = 0.0;
		for (int j = 0; j < FEATURE_DIM; j++) {
			norm += points[i].features[j] * points[i].features[j];
		}
		row_norms[i] = norm;
	}

	// Compute pairwise distances using the dot product and the optimized formula
	for (int i = 0; i < num_pts; i++) {
		for (int j = i; j < num_pts; j++) {
			if (i == j) {
				dist_matrix[i][j] = 0.0; // Distance from a point to itself is 0.0
				continue;
			}

			double dot_product = 0.0;
			for (int k = 0; k < FEATURE_DIM; k++) {
				dot_product += points[i].features[k] * points[j].features[k];
			}

			// Distance formula: sqrt(||A||^2 + ||B||^2 - 2 * A.B)
			double distance = sqrt(row_norms[i] + row_norms[j] - 2.0 * dot_product);
			dist_matrix[i][j] = distance;
			dist_matrix[j][i] = distance; // Symmetric matrix
		}
	}

	// Free temporary memory
	free(row_norms);

	return dist_matrix;
}

/*------------------------------------------------------
  Get the index of the centroid from centroid array, which has samallest distance to given point.
	POINT* pt		     : point in euclidian space with FEATURE_DIM features.
	POINT* centroid_arr  : array of points representing the centroids.
	int n_cluster		 : number of clusters ( = number of centroids)
  Output:
	int clusterIndex: index of the centroid in cent_arr
------------------------------------------------------*/
int get_index_of_nearest_centroid(POINT* pt, POINT* centroid_arr, int n_cluster)
{
	int i, clusterIndex;
	double d, min_d;

	min_d = HUGE_VAL;
	clusterIndex = pt->group;
	for (i = 0; i < n_cluster; i++) {
		d = euc_dist_squared(&centroid_arr[i], pt);
		if (d < min_d) {
			min_d = d;
			clusterIndex = i;
		}
	}
	return clusterIndex;
}

/*------------------------------------------------------
 Get the distance to the closest centroid
  Input:
	POINT* pt			 : point in euclidian space with FEATURE_DIM features.
	POINT* centroid_arr  : array of points representing the centroids.
	int n_cluster		 : number of clusters ( = number of centroids)
  Output:
	double min_d: distnace to the closest centroid
------------------------------------------------------*/
double get_distance_to_nearest_centroid(POINT* pt, POINT* centroid_arr, int n_cluster)
{
	int i;
	double d, min_d;

	min_d = HUGE_VAL;
	for (i = 0; i < n_cluster; i++) {
		d = euc_dist_squared(&centroid_arr[i], pt);
		if (d < min_d) {
			min_d = d;
		}
	}

	return min_d;
}


/*-------------------------------------------------------
	Centroid init with kmeans++ algorithm:
		randomly choose point form data for the first cluster
		Keep an array of "minimal distances" which will store for each point the distance only to the nearest centroid
		Keep a sum of "minimal distances" to use later with probability
		For each new centroid left to init:
			for each point save the minima distance to the "winner" centroid and add this distance to the total sum distance.
			Randomly take proportional part of the total sum
			For each point subtract its minimum distance from the proportional total sum - if the distance is large node has higher chance of zeroing the total sum (this is the node we want for next centroid)
		Update the group attribute of all points to new cluster index.

		INPUT:
			POINT* pts: array of all the points in the data
			int num_pts: length of pts array
			POINT* centroids: array to store the points which will be the centroids
			int num_clusters: length of centroids array
		Output:
			No direct output
			Update the parameters:
				POINT* pts: array of all the points in the data.
				POINT* centroids: array of points storing the centroids.
-------------------------------------------------------*/
void init_clusters_with_kpp(POINT* pts, int num_pts, POINT* centroids, int num_clusters)
{
	int j; // Index
	int cluster; // index
	double sum; // sum of minimal distances
	double* distances; // array of distances between each point and its closest centroid

	distances = (double*)malloc(sizeof(double) * num_pts);

	// randomly choose point form data for the first cluster
	centroids[0] = pts[rand() % num_pts];

	/* Select the centroids for the remaining clusters. */
	for (cluster = 1; cluster < num_clusters; cluster++) {

		/* For each data point find the nearest centroid, save its
		   distance in the distance array, then add it to the sum of
		   total distance. */
		sum = 0.0;
		for (j = 0; j < num_pts; j++) {
			distances[j] = get_distance_to_nearest_centroid(&pts[j], centroids, cluster);
			sum += distances[j];
		}

		/* Get a random number from 0 to sum_of_all_distances. */
		sum = sum * rand() / (RAND_MAX - 1);

		/* Assign the centroids. the point with the largest distance
			will have a greater probability of being selected. */
		for (j = 0; j < num_pts; j++) {
			sum -= distances[j];
			if (sum <= 0)
			{
				centroids[cluster] = pts[j];
				break;
			}
		}
	}

	/* Assign each observation the index of it's nearest cluster centroid. */
	for (j = 0; j < num_pts; j++)
		pts[j].group = get_index_of_nearest_centroid(&pts[j], centroids, num_clusters);

	free(distances);

	return;
}	/* end, kpp */

/*-------------------------------------------------------
	Lloyd's K-Means algorithm (the most popular version) with Kmeans++ centroid initialization:
		Initializing the centroids using kmeans++ and clustering the points.
		Iterating until conversion (no points move between clusters):
			zero the centroids
			sum points in culsets elementwise (phase 1 of avarage calcualtion)
			devide points in cluster element wise (phase 2 of avarage calcualtion)
			update centroids to be the avarage location
			check if points change clusters

	INPUT:
		POINT* pts: point array to cluster by updating the group atribute
		int num_pts: length of points
		int num_clusters: number of clusters (k)
	OUTPUT:
		POINT* centroids: array of points represeting the centroids
		Second output(with parameter): POINT* pts - array of points with updated groups (clusters)

-------------------------------------------------------*/
POINT* kmeans_plus_plus(POINT* pts, int num_pts, int num_clusters)
{
	int i, clusterIndex; // indexes
	int changes; // number of changes

	// Check input
	if (num_clusters == 1 || num_pts <= 0 || num_clusters > num_pts)
		return 0;

	// Init the centroid array
	POINT* centroids = (POINT*)malloc(sizeof(POINT) * num_clusters);

	// Get first k centrodis
	init_clusters_with_kpp(pts, num_pts, centroids, num_clusters);

	do {
		// Zero all centroids attributes - we need to sum elementwise all the points starting from zero
		for (i = 0; i < num_clusters; i++) {
			centroids[i].group = 0;
			for (int j = 0; j < FEATURE_DIM; j++) {
				centroids[i].features[j] = 0;
			}
		}

		// Sum all points elementwise. store the sum of each feature in the centroid point (in the corresponding feature)
		for (i = 0; i < num_pts; i++) {
			clusterIndex = pts[i].group;		// get the centroid index (which is also the cluster number ) of the point from the pervoius iteration
			centroids[clusterIndex].group++;	// Update centroid's group attribute - which is a counter of points in the cluster at this time
			for (int j = 0; j < FEATURE_DIM; j++) {
				centroids[clusterIndex].features[j] += pts[i].features[j]; // add the data in the feature of the point to the corresponding summation stored in the centroid
			}
		}

		// Get the avarage of all points in cluster using deviation
		for (i = 0; i < num_clusters; i++) {
			if (centroids[i].group > 0) {
				for (int j = 0; j < FEATURE_DIM; j++) {
					// Devide each feature of the centroid (which stores the sum of that feature for points int the cluster) by the count of points in that cluster (stored in centroidd's group)
					centroids[i].features[j] /= centroids[i].group;
				}
			}
		}

		// Reassign points to nearest centroid
		changes = 0;
		for (i = 0; i < num_pts; i++) {
			clusterIndex = get_index_of_nearest_centroid(&pts[i], centroids, num_clusters); // get the new distance of each point and its centroid
			// Check if nearest centroid changed
			if (clusterIndex != pts[i].group) {
				pts[i].group = clusterIndex;
				changes++;
			}
		}


		printf("changes: %d\n", changes);
	} while ((changes > 0)); // there is no point to run the loop until 100% conversion

	// Set each centroid's group index
	for (i = 0; i < num_clusters; i++)
		centroids[i].group = i;

	return centroids;
}


/*------------------------------------------------------
 Calculate silhouette score for a single point
  Input:
	POINT* pts             : array of all points in the dataset
	int    num_pts         : number of points in the dataset
	int    point_index     : index of the point for which to calculate the silhouette score
	int    k               : number of clusters
	double** distance_matrix : precomputed pairwise distance matrix for all points
  Output:
	double silhouette_score_for_point : silhouette score for the given point

   The valid silhouette score is [-1,1]. In case of error in the code, return NAN.
------------------------------------------------------*/
double calculate_silhouette_for_point(POINT* pts, int num_pts, int point_index, int k, double** distance_matrix) {
	double silhouette_score_for_point = 0.0;
	double a = 0.0;
	double b = HUGE_VAL;
	double avg_distance = 0.0;
	int count_same_cluster_points = 0;
	double sum_dist_same_cluster = 0.0;
	double* sum_dist_other_clusters = NULL;
	int* count_other_clusters_points = NULL;

	// Memory allocation for the distance array from the point to other clusters
	sum_dist_other_clusters = (double*)calloc(k, sizeof(double));
	if (sum_dist_other_clusters == NULL) {
		printf("Memory allocation failed for sum_dist_other_clusters\n");
		return NAN; // NAN - Not-a-Nuber
	}

	// Memory allocation for points count in each other cluster
	count_other_clusters_points = (int*)calloc(k, sizeof(int));
	if (count_other_clusters_points == NULL) {
		printf("Memory allocation failed for count_other_clusters_points\n");
		free(sum_dist_other_clusters); // Free previously allocated memory
		return NAN; // NAN - Not-a-Nuber
	}

	// Calculate the silhouette for point point_index => the cluster of the point is pts[point_index].group

	for (int i = 0; i < num_pts; i++) {
		// Calculate a - cohesion
		if ((pts[point_index].group == pts[i].group) && (i != point_index)) { // If the points in the same cluster
			sum_dist_same_cluster += distance_matrix[point_index][i];
			count_same_cluster_points++;
		}

		// Calculate b - separation
		if ((pts[point_index].group != pts[i].group) && (i != point_index)) {
			sum_dist_other_clusters[pts[i].group] += distance_matrix[point_index][i];
			count_other_clusters_points[pts[i].group]++;
		}
	}
	// Calculate a - cohesion
	a = sum_dist_same_cluster / count_same_cluster_points;

	// Calculate b - separation
	for (int i = 0; i < k; i++) {
		if (count_other_clusters_points[i] > 0) {
			avg_distance = sum_dist_other_clusters[i] / count_other_clusters_points[i];
			b = fminf(avg_distance, b);
		}
	}

	// Free temporary allocated memory
	free(sum_dist_other_clusters);
	free(count_other_clusters_points);


	silhouette_score_for_point = (b - a) / fmax(b, a);

	return silhouette_score_for_point;
}

/*------------------------------------------------------
 Calculate average silhouette score for all points
  Input:
	POINT* pts            : array of all points in the dataset
	int    num_pts        : number of points in the dataset
	int    k              : number of clusters
	double** distance_matrix : precomputed pairwise distance matrix for all points
  Output:
	double silhouette_score : average silhouette score for the dataset

  The valid silhouette score is [-1,1]. In case of error in the code, return NAN.
------------------------------------------------------*/
double calculate_silhouette_score(POINT* pts, int num_pts, int k, double** distance_matrix) {

	double silhouette_score = 0.0;
	double sum_silhouette = 0.0;
	double sum_calculation = 0.0;

	// Iterate over each point to calculate its silhouette score
	for (int i = 0; i < num_pts; i++) {
		// Calculate silhouette score for point i
		sum_calculation = calculate_silhouette_for_point(pts, num_pts, i, k, distance_matrix);

		// Check if the result is NaN - error in silhouette calclulation for point i
		if (isnan(sum_calculation)) {
			return NAN;
		}

		// Add silhouette score for point i to the total
		sum_silhouette += sum_calculation;
	}

	silhouette_score = sum_silhouette / num_pts;
	return silhouette_score;
}


/*-------------------------------------------------------
	This function prints all given points and centroids
	INPUT:
		POINT* pts: Array of Points - all the points in the dataset
		int num_pts: length of pts array
		POINT* centroids: array of Points - only the centroids
		int num_clusters: length of centrodis array
-------------------------------------------------------*/
void print_points_and_centroids(POINT* pts, int num_pts, POINT* centroids, int num_clusters)
{
	// Print point details
	for (int i = 0; i < num_pts; i++) {
		printf("Point %d - Cluster: %d, Features: ", i, pts[i].group);

		// Print first few features to avoid overwhelming output
		for (int j = 0; j < 5; j++) {
			printf("%.2f ", pts[i].features[j]);
		}
		printf("...\n");
	}

	// Print centroid details
	printf("\nCentroids:\n");
	for (int i = 0; i < num_clusters; i++) {
		printf("Cluster %d - Features: ", i);
		for (int j = 0; j < 5; j++) {
			printf("%.2f ", centroids[i].features[j]);
		}
		printf("...\n");
	}
}

/*-------------------------------------------------------
	This function frees the memory for a distance matrix
	INPUT:
		double** distance_matrix: precomputed pairwise distance matrix for all points
		int num_pts: number of points in the dataset
-------------------------------------------------------*/
void free_distance_matrix(double** distance_matrix, int num_pts) {
	for (int i = 0; i < num_pts; i++) {
		free(distance_matrix[i]);
	}
	free(distance_matrix);
}

/*-------------------------------------------------------
	main
-------------------------------------------------------*/
int main()
{
	int num_pts;
	double silhouette_score = 0.0;

	clock_t start, end;
	double cpu_time_used;

	start = clock();

	POINT* pts = read_points_from_csv("C:/Users/Oren Baranovsky/source/repos/kmeans_plusplus_siluette/kmeans_plusplus_siluette/kmeans_data.csv", &num_pts);
	if (!pts) {
		printf("Failed to read points\n");
		return 1;
	}

	// Make sure its Dans database of 10K vectors
	if (num_pts != 10000)
	{
		printf("Check dataset - expected 10K vectors\n");
		return 1;
	}

	int num_clusters = NUMBER_OF_CLUSTERS;
	int maxTimes = MAXIMUM_ITERATIONS;


	// Pre-calculate distances from point to every other point
	double** dist_matrix = pairwise_distances(pts, num_pts);
	if (!dist_matrix) {
		printf("Failed to pre-calculate distances\n");
		free(pts);
		return 1;
	}

	for (int i = 2; i <= num_clusters; i++) {

		POINT* centroids = kmeans_plus_plus(pts, num_pts, i);

		//print_points_and_centroids(pts, num_pts, centroids, i);


		silhouette_score = calculate_silhouette_score(pts, num_pts, i, dist_matrix);
		if (isnan(silhouette_score)) {
			printf("Failed to calculate the silhouette score\n");
			free(pts);
			free(centroids);
			return 1;
		}


		printf("Silhouette score for round %d is: %f\n", i, silhouette_score);
		free(centroids);
	}

	free(pts);
	free_distance_matrix(dist_matrix, num_pts);

	end = clock();
	cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

	printf("Time cpu: %f seconds", cpu_time_used);

	return 0;
}