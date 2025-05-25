#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>
#include <map>

using namespace std;

const int image_width = 28;
const int image_height = 28;
const double min_value = 0.0;
const double max_value = 255.0;

struct Point {
    vector<double> features;
    int label;
    
    Point() : label(-1) {}
    Point(const vector<double>& f, int l = -1) : features(f), label(l) {}
};

struct Centroid {
    vector<double> center;
    int count;
    
    Centroid() : count(0) {}
    Centroid(int dim) : center(dim, 0.0), count(0) {}
    Centroid(const vector<double>& c) : center(c), count(0) {}
};

class FederatedKMeans {
private:
    int rank, size;
    int k;  // number of clusters
    int dimensions;
    int max_iterations;
    double tolerance;
    vector<Point> local_data;
    vector<Centroid> global_centroids;
    vector<Centroid> local_centroids;
    
public:
    FederatedKMeans(int k_clusters, int max_iter = 100, double tol = 1e-6) 
                    : k(k_clusters), max_iterations(max_iter), tolerance(tol) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    
    // Load and distribute data with heterogeneity simulation
    void loadData(const string& filename) {
        vector<Point> all_data;
        
        if (rank == 0) {
            // Server loads all data
            ifstream file(filename);
            string line;
            
            while (getline(file, line)) {
                istringstream iss(line);
                vector<double> features;
                string pixel_value;
                
                while (getline(iss, pixel_value, ',')) {
                    try {
                        features.push_back(stod(pixel_value));
                    } catch (...) {
                        continue;
                    }
                }
                
                if (!features.empty()) {
                    normalizeFeatures(features);
                    all_data.push_back(Point(features));
                }
            }
            
            if (!all_data.empty()) {
                dimensions = all_data[0].features.size();
                cout << "Loaded " << all_data.size() << " points with " 
                         << dimensions << " dimensions" << endl;
            }
            
            // Simulate data heterogeneity by applying different transformations
            distributeHeterogeneousData(all_data);
        } else {
            // Workers receive their portion
            receiveLocalData();
        }
    }
    
    // Normalize pixel values from 0-255 to 0-1 range
    void normalizeFeatures(vector<double>& features) {
        for (double& f : features) {
            f = (f - min_value) / (max_value - min_value);
            // Clamp to [0, 1] range
            f = max(0.0, min(1.0, f));
        }
    }

    void distributeHeterogeneousData(const vector<Point>& all_data) {
        int total_points = all_data.size();
        int workers = size - 1;  // Exclude server
        
        // Send dimensions to all workers
        for (int i = 1; i < size; i++) {
            MPI_Send(&dimensions, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        
        // Distribute data with heterogeneity
        for (int worker = 1; worker < size; worker++) {
            int start_idx = (worker - 1) * total_points / workers;
            int end_idx = worker * total_points / workers;
            int local_size = end_idx - start_idx;
            
            // Send local data size
            MPI_Send(&local_size, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            
            // Apply transformation based on worker ID to simulate heterogeneity
            vector<Point> transformed_data;
            for (int i = start_idx; i < end_idx; i++) {
                Point p = all_data[i];
                applyTransformation(p, worker);
                transformed_data.push_back(p);
            }
            
            // Send transformed data
            for (const auto& point : transformed_data) {
                MPI_Send(point.features.data(), dimensions, MPI_DOUBLE, worker, 2, MPI_COMM_WORLD);
            }
        }
    }
    
    void applyTransformation(Point& point, int worker_id) {
        vector<double> transformed_image = point.features;

        // Apply different transformations to simulate data heterogeneity
        switch (worker_id % 4) {
            case 1: // 90-degree rotation
                transformed_image = rotateImage90(transformed_image);
                break;
                
            case 2: // 180-degree rotation
                transformed_image = rotateImage180(transformed_image);
                break;
                
            case 3: // 270-degree rotation
                transformed_image = rotateImage270(transformed_image);
                break;
            default: // No transformation
                break;
        }
    }
    
    // Rotate image 90 degrees clockwise
    vector<double> rotateImage90(const vector<double>& image) {
        vector<double> rotated(image.size());
        for (int y = 0; y < image_height; y++) {
            for (int x = 0; x < image_width; x++) {
                int old_idx = y * image_width + x;
                int new_x = image_height - 1 - y;
                int new_y = x;
                int new_idx = new_y * image_width + new_x;
                rotated[new_idx] = image[old_idx];
            }
        }
        return rotated;
    }
    
    // Rotate image 180 degrees
    vector<double> rotateImage180(const vector<double>& image) {
        vector<double> rotated(image.size());
        for (int y = 0; y < image_height; y++) {
            for (int x = 0; x < image_width; x++) {
                int old_idx = y * image_width + x;
                int new_x = image_width - 1 - x;
                int new_y = image_height - 1 - y;
                int new_idx = new_y * image_width + new_x;
                rotated[new_idx] = image[old_idx];
            }
        }
        return rotated;
    }
    
    // Rotate image 270 degrees clockwise
    vector<double> rotateImage270(const vector<double>& image) {
        vector<double> rotated(image.size());
        for (int y = 0; y < image_height; y++) {
            for (int x = 0; x < image_width; x++) {
                int old_idx = y * image_width + x;
                int new_x = y;
                int new_y = image_width - 1 - x;
                int new_idx = new_y * image_width + new_x;
                rotated[new_idx] = image[old_idx];
            }
        }
        return rotated;
    }

    void receiveLocalData() {
        // Receive dimensions
        MPI_Recv(&dimensions, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive local data size
        int local_size;
        MPI_Recv(&local_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive local data
        local_data.resize(local_size);
        for (int i = 0; i < local_size; i++) {
            vector<double> features(dimensions);
            MPI_Recv(features.data(), dimensions, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_data[i] = Point(features);
        }
        
        cout << "Worker " << rank << " received " << local_size << " points" << endl;
    }
    
    void initialiseCentroids() {
        if (rank == 0) {
            // Server Initialises random centroids
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(-1.0, 1.0);
            
            global_centroids.resize(k);
            for (int i = 0; i < k; i++) {
                global_centroids[i].center.resize(dimensions);
                for (int j = 0; j < dimensions; j++) {
                    global_centroids[i].center[j] = dis(gen);
                }
            }
            
            // Broadcast initial centroids to all workers
            broadcastCentroids();
        } else {
            // Workers receive initial centroids
            receiveCentroids();
        }
    }
    
    void broadcastCentroids() {
        for (int worker = 1; worker < size; worker++) {
            for (int i = 0; i < k; i++) {
                MPI_Send(global_centroids[i].center.data(), dimensions, 
                        MPI_DOUBLE, worker, 3, MPI_COMM_WORLD);
            }
        }
    }
    
    void receiveCentroids() {
        global_centroids.resize(k);
        for (int i = 0; i < k; i++) {
            global_centroids[i].center.resize(dimensions);
            MPI_Recv(global_centroids[i].center.data(), dimensions, 
                    MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
    double euclideanDistance(const vector<double>& a, const vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    
    void localKMeansStep() {
        // Initialise local centroids
        local_centroids.assign(k, Centroid(dimensions));
        
        // Assign points to clusters and update local centroids
        for (auto& point : local_data) {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = 0;
            
            // Find closest centroid
            for (int i = 0; i < k; i++) {
                double dist = euclideanDistance(point.features, global_centroids[i].center);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = i;
                }
            }
            
            // Update local centroid
            point.label = best_cluster;
            local_centroids[best_cluster].count++;
            for (int j = 0; j < dimensions; j++) {
                local_centroids[best_cluster].center[j] += point.features[j];
            }
        }
        
        // Compute average for each local centroid
        for (int i = 0; i < k; i++) {
            if (local_centroids[i].count > 0) {
                for (int j = 0; j < dimensions; j++) {
                    local_centroids[i].center[j] /= local_centroids[i].count;
                }
            }
        }
    }
    
    void federatedAveraging() {
        if (rank == 0) {
            // Server collects local centroids and performs federated averaging
            vector<Centroid> aggregated_centroids(k, Centroid(dimensions));
            vector<int> total_counts(k, 0);
            
            // Collect from all workers
            for (int worker = 1; worker < size; worker++) {
                for (int i = 0; i < k; i++) {
                    vector<double> worker_centroid(dimensions);
                    int worker_count;
                    
                    MPI_Recv(worker_centroid.data(), dimensions, MPI_DOUBLE, 
                            worker, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&worker_count, 1, MPI_INT, worker, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    total_counts[i] += worker_count;
                    if (worker_count > 0) {
                        for (int j = 0; j < dimensions; j++) {
                            aggregated_centroids[i].center[j] += worker_centroid[j] * worker_count;
                        }
                    }
                }
            }
            
            // Compute weighted average
            for (int i = 0; i < k; i++) {
                if (total_counts[i] > 0) {
                    for (int j = 0; j < dimensions; j++) {
                        global_centroids[i].center[j] = aggregated_centroids[i].center[j] / total_counts[i];
                    }
                }
            }
            
            // Broadcast updated centroids
            broadcastCentroids();
        } else {
            // Workers send their local centroids
            for (int i = 0; i < k; i++) {
                MPI_Send(local_centroids[i].center.data(), dimensions, 
                        MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
                MPI_Send(&local_centroids[i].count, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
            }
            
            // Receive updated centroids
            receiveCentroids();
        }
    }
    
    double computeLocalInertia() {
        double inertia = 0.0;
        for (const auto& point : local_data) {
            if (point.label >= 0 && point.label < k) {
                double dist = euclideanDistance(point.features, global_centroids[point.label].center);
                inertia += dist * dist;
            }
        }
        return inertia;
    }
    
    double computeGlobalInertia() {
        double local_inertia = computeLocalInertia();
        double global_inertia = 0.0;
        
        MPI_Reduce(&local_inertia, &global_inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        return global_inertia;
    }
    
    void train() {
        initialiseCentroids();
        
        double prev_inertia = numeric_limits<double>::max();
        
        for (int iteration = 0; iteration < max_iterations; iteration++) {
            // Local k-means step
            localKMeansStep();
            
            // Federated averaging
            federatedAveraging();
            
            double current_inertia = computeGlobalInertia();
            
            if (rank == 0) {
                cout << "Iteration " << iteration + 1 << ", Inertia: " 
                         << fixed << setprecision(6) << current_inertia << endl;
                
                // Check convergence
                if (abs(prev_inertia - current_inertia) < tolerance) {
                    cout << "Converged after " << iteration + 1 << " iterations" << endl;
                    break;
                }
            }
            
            prev_inertia = current_inertia;
        }
    }
    
    void printResults() {
        if (rank == 0) {
            cout << "\nFinal Centroids:" << endl;
            for (int i = 0; i < k; i++) {
                cout << "Cluster " << i << ": ";
                for (int j = 0; j < dimensions; j++) {
                    cout << fixed << setprecision(4) 
                             << global_centroids[i].center[j] << " ";
                }
                cout << endl;
            }
        }
        
        // Each worker prints local statistics
        map<int, int> local_cluster_counts;
        for (const auto& point : local_data) {
            local_cluster_counts[point.label]++;
        }
        
        cout << "\nWorker " << rank << " cluster assignments:" << endl;
        for (const auto& pair : local_cluster_counts) {
            cout << "  Cluster " << pair.first << ": " << pair.second << " points" << endl;
        }
    }
};

// Centralised K-Means for comparison
class CentralisedKMeans {
private:
    int k;
    int dimensions;
    int max_iterations;
    double tolerance;
    vector<Point> data;
    vector<Centroid> centroids;
    
public:
    CentralisedKMeans(int k_clusters, int max_iter = 100, double tol = 1e-6) 
                      : k(k_clusters), max_iterations(max_iter), tolerance(tol) {}
    
    void loadData(const string& filename) {
        ifstream file(filename);
        string line;
        
        while (getline(file, line)) {
            istringstream iss(line);
            vector<double> features;
            string pixel_value;
            
            while (getline(iss, pixel_value, ',')) {
                try {
                    features.push_back(stod(pixel_value));
                } catch (...) {
                    continue;
                }
            }
            
            if (!features.empty()) {
                normalizeFeatures(features);
                data.push_back(Point(features));
            }
        }
        
        if (!data.empty()) {
            dimensions = data[0].features.size();
            cout << "Centralised: Loaded " << data.size() << " points with " 
                     << dimensions << " dimensions" << endl;
        }
    }
    
    // Normalize pixel values from 0-255 to 0-1 range
    void normalizeFeatures(vector<double>& features) {
        for (double& f : features) {
            f = (f - min_value) / (max_value - min_value);
            // Clamp to [0, 1] range
            f = max(0.0, min(1.0, f));
        }
    }

    double euclideanDistance(const vector<double>& a, const vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    
    void train() {
        // Initialise centroids randomly
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1.0, 1.0);
        
        centroids.resize(k);
        for (int i = 0; i < k; i++) {
            centroids[i].center.resize(dimensions);
            for (int j = 0; j < dimensions; j++) {
                centroids[i].center[j] = dis(gen);
            }
        }
        
        double prev_inertia = numeric_limits<double>::max();
        
        for (int iteration = 0; iteration < max_iterations; iteration++) {
            // Assignment step
            for (auto& point : data) {
                double min_dist = numeric_limits<double>::max();
                int best_cluster = 0;
                
                for (int i = 0; i < k; i++) {
                    double dist = euclideanDistance(point.features, centroids[i].center);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = i;
                    }
                }
                point.label = best_cluster;
            }
            
            // Update step
            for (int i = 0; i < k; i++) {
                fill(centroids[i].center.begin(), centroids[i].center.end(), 0.0);
                centroids[i].count = 0;
            }
            
            for (const auto& point : data) {
                int cluster = point.label;
                centroids[cluster].count++;
                for (int j = 0; j < dimensions; j++) {
                    centroids[cluster].center[j] += point.features[j];
                }
            }
            
            for (int i = 0; i < k; i++) {
                if (centroids[i].count > 0) {
                    for (int j = 0; j < dimensions; j++) {
                        centroids[i].center[j] /= centroids[i].count;
                    }
                }
            }
            
            // Compute inertia
            double inertia = 0.0;
            for (const auto& point : data) {
                double dist = euclideanDistance(point.features, centroids[point.label].center);
                inertia += dist * dist;
            }
            
            cout << "Centralised Iteration " << iteration + 1 << ", Inertia: " 
                     << fixed << setprecision(6) << inertia << endl;
            
            if (abs(prev_inertia - inertia) < tolerance) {
                cout << "Centralised converged after " << iteration + 1 << " iterations" << endl;
                break;
            }
            
            prev_inertia = inertia;
        }
    }
    
    void printResults() {
        cout << "\nCentralised Final Centroids:" << endl;
        for (int i = 0; i < k; i++) {
            cout << "Cluster " << i << ": ";
            for (int j = 0; j < dimensions; j++) {
                cout << fixed << setprecision(4) 
                         << centroids[i].center[j] << " ";
            }
            cout << endl;
        }
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 3) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <data_file> <k_clusters>" << endl;
            cerr << "Example: " << argv[0] << " data.csv 3" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    string data_file = argv[1];
    int k = stoi(argv[2]);
    
    if (size < 3) {
        if (rank == 0) {
            cerr << "This program requires at least 3 processes (1 server + 2 workers)" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        cout << "=== Federated K-Means Clustering ===" << endl;
        cout << "Processes: " << size << " (1 server + " << (size-1) << " workers)" << endl;
        cout << "Data file: " << data_file << endl;
        cout << "K clusters: " << k << endl;
        cout << endl;
    }
    
    // Federated Learning
    FederatedKMeans fed_kmeans(k);
    fed_kmeans.loadData(data_file);
    
    double start_time = MPI_Wtime();
    fed_kmeans.train();
    double fed_time = MPI_Wtime() - start_time;
    
    if (rank == 0) {
        cout << "\nFederated training time: " << fed_time << " seconds" << endl;
    }
    
    fed_kmeans.printResults();
    
    // Centralised comparison
    if (rank == 0) {
        cout << "\n=== Centralised K-Means (Baseline) ===" << endl;
        CentralisedKMeans cent_kmeans(k);
        cent_kmeans.loadData(data_file);
        
        start_time = MPI_Wtime();
        cent_kmeans.train();
        double cent_time = MPI_Wtime() - start_time;
        
        cout << "Centralised training time: " << cent_time << " seconds" << endl;
        cent_kmeans.printResults();
        
        cout << "\n=== Performance Comparison ===" << endl;
        cout << "Federated time: " << fed_time << "s" << endl;
        cout << "Centralised time: " << cent_time << "s" << endl;
        cout << "Speedup: " << cent_time / fed_time << "x" << endl;
    }
    
    MPI_Finalize();
    return 0;
}