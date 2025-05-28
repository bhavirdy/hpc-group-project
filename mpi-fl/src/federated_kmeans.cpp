#include <mpi.h>
#include <unistd.h>
#include <limits.h>   
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>
#include <map>
#include <filesystem>

using namespace std;

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
    int dimensions = 100;
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
    
    vector<string> findDataFiles(const string& directory = "./data/uci_har/processed_data/split_data") {
        vector<string> files;
        if (!filesystem::exists(directory)) {
            if (rank == 0) {
                cout << "Directory does not exist: " << directory << endl;
            }
        }
        else {
            for (const auto& entry : filesystem::directory_iterator(directory)) {
                if (entry.path().extension() == ".csv") {
                    files.push_back(entry.path().string());
                }
            }
        }
        sort(files.begin(), files.end());
        return files;
    }
    
    void loadData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Worker " << rank << " could not open: " << filename << endl;
            return;
        }
        
        string line;
        while (getline(file, line)) {
            if (line.empty()) continue;
            
            istringstream iss(line);
            vector<double> features;
            string value;
            
            while (getline(iss, value, ',')) {
                try {
                    features.push_back(stod(value));
                } catch (...) {
                    continue;
                }
            }
            
            if (!features.empty()) {
                if (dimensions == 0) {
                    dimensions = features.size();
                }
                if (features.size() == dimensions) {
                    local_data.push_back(Point(features));
                }
            }
        }
        
        cout << "Worker " << rank << " loaded " << local_data.size() << " points" << endl;
    }
    
    void distributeData() {
        if (rank == 0) {
            // Server distributes files
            vector<string> files = findDataFiles();
            int num_workers = size - 1;
            
            cout << "Distributing " << files.size() << " files to " << num_workers << " workers" << endl;
            
            // Round-robin distribution
            for (int i = 0; i < files.size(); i++) {
                int worker = (i % num_workers) + 1;
                string filename = files[i];
                int len = filename.length();

                cout << "Sending " << filename << " to worker " << worker << endl;
                
                MPI_Send(&len, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                MPI_Send(filename.c_str(), len, MPI_CHAR, worker, 1, MPI_COMM_WORLD);
            }
            
            // Send termination signal
            for (int worker = 1; worker < size; worker++) {
                int len = 0;
                MPI_Send(&len, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            }
            
        } else {
            // Workers receive and load their files
            while (true) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if (len == 0) break; // No more files
                
                char* buffer = new char[len + 1];
                MPI_Recv(buffer, len, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                buffer[len] = '\0';

                string filename(buffer);
                cout << "Worker " << rank << " received filename: " << filename << endl;
                loadData(string(buffer));
                delete[] buffer;
            }
        }
    }

    void initialiseCentroids() {
        if (rank == 0) {
            // Server initializes random centroids
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(-50.0, 50.0);
            
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
        // Broadcast dimensions first
        for (int worker = 1; worker < size; worker++) {
            MPI_Send(&dimensions, 1, MPI_INT, worker, 16, MPI_COMM_WORLD);
        }
        
        // Broadcast centroids
        for (int worker = 1; worker < size; worker++) {
            for (int i = 0; i < k; i++) {
                MPI_Send(global_centroids[i].center.data(), dimensions, 
                        MPI_DOUBLE, worker, 3, MPI_COMM_WORLD);
            }
        }
    }
    
    void receiveCentroids() {
        // Receive dimensions (in case worker doesn't have data)
        MPI_Recv(&dimensions, 1, MPI_INT, 0, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
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
        // Initialize local centroids
        local_centroids.assign(k, Centroid(dimensions));
        
        // Skip if no local data (idle worker)
        if (local_data.empty()) {
            return;
        }
        
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
            
            // Collect from all workers (including idle ones)
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
        } 
        else {
            // Workers send their local centroids (even idle workers send zeros)
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
        if (!local_data.empty()) {
            map<int, int> local_cluster_counts;
            for (const auto& point : local_data) {
                local_cluster_counts[point.label]++;
            }
            
            cout << "\nWorker " << rank << " cluster assignments:" << endl;
            for (const auto& pair : local_cluster_counts) {
                cout << "  Cluster " << pair.first << ": " << pair.second << " points" << endl;
            }
        } else if (rank != 0) {
            cout << "\nWorker " << rank << " had no data (idle)" << endl;
        }
    }
};

// Centralized K-Means for comparison
class CentralizedKMeans {
private:
    int k;
    int dimensions;
    int max_iterations;
    double tolerance;
    vector<Point> data;
    vector<Centroid> centroids;
    
public:
    CentralizedKMeans(int k_clusters, int max_iter = 100, double tol = 1e-6) 
                      : k(k_clusters), max_iterations(max_iter), tolerance(tol) {}
    
    void loadData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Could not open " << filename << " for centralized comparison" << endl;
            return;
        }
        
        string line;
        
        while (getline(file, line)) {
            if (line.empty()) continue;
            
            istringstream iss(line);
            vector<double> features;
            string value;
            
            while (getline(iss, value, ',')) {
                try {
                    features.push_back(stod(value));
                } catch (...) {
                    continue;
                }
            }
            
            if (!features.empty()) {
                data.push_back(Point(features));
            }
        }
        
        if (!data.empty()) {
            dimensions = data[0].features.size();
            cout << "Centralized: Loaded " << data.size() << " points with " 
                     << dimensions << " dimensions" << endl;
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
        if (data.empty()) return;
        
        // Initialize centroids randomly
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-50.0, 50.0);
        
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
            
            cout << "Centralized Iteration " << iteration + 1 << ", Inertia: " 
                     << fixed << setprecision(6) << inertia << endl;
            
            if (abs(prev_inertia - inertia) < tolerance) {
                cout << "Centralized converged after " << iteration + 1 << " iterations" << endl;
                break;
            }
            
            prev_inertia = inertia;
        }
    }
    
    void printResults() {
        cout << "\nCentralized Final Centroids:" << endl;
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

    // Get the hostname of the node this process is running on
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);

    // Print which node the process is running on
    std::cout << "Process " << rank << " running on node " << hostname << std::endl;
    
    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <k_clusters>" << endl;
            cerr << "Example: " << argv[0] << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int k = stoi(argv[1]);
    string centralized_file = "./data/uci_har/processed_data/X_train_pca.csv";
    
    if (size < 2) {
        if (rank == 0) {
            cerr << "This program requires at least 2 processes (1 server + 1 worker)" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        cout << "=== Federated K-Means Clustering ===" << endl;
        cout << "Processes: " << size << " (1 server + " << (size-1) << " workers)" << endl;
        cout << "K clusters: " << k << endl;
        cout << endl;
    }
    
    // Federated Learning
    FederatedKMeans fed_kmeans(k);
    fed_kmeans.distributeData();
    
    double start_time = MPI_Wtime();
    fed_kmeans.train();
    double fed_time = MPI_Wtime() - start_time;
    
    if (rank == 0) {
        cout << "\nFederated training time: " << fed_time << " seconds" << endl;
    }
    
    //fed_kmeans.printResults();
        
    // Centralized comparison
    if (rank == 0) {
        cout << "\n=== Centralized K-Means ===" << endl;
        CentralizedKMeans cent_kmeans(k);
        cent_kmeans.loadData(centralized_file);
        
        start_time = MPI_Wtime();
        cent_kmeans.train();
        double cent_time = MPI_Wtime() - start_time;
        
        cout << "Centralized training time: " << cent_time << " seconds" << endl;
        //cent_kmeans.printResults();
        
        cout << "\n=== Performance Comparison ===" << endl;
        cout << "Federated time: " << fed_time << "s" << endl;
        cout << "Centralized time: " << cent_time << "s" << endl;
    }
    
    MPI_Finalize();
    return 0;
}