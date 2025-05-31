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
    string source_file;  // Track which file this point came from

    Point() : label(-1) {}
    Point(const vector<double>& f, int l = -1, const string& src = "") : features(f), label(l), source_file(src) {}
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
    int k;
    int dimensions = 100;
    int max_iterations;
    double tolerance;
    map<string, vector<Point>> file_to_data;
    vector<Point> test_data;  // Global test data loaded by master
    vector<Centroid> global_centroids;
    vector<Centroid> local_centroids;

public:
    FederatedKMeans(int k_clusters, int max_iter = 100, double tol = 1e-6)
        : k(k_clusters), max_iterations(max_iter), tolerance(tol) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    vector<string> findDataFiles(const string& directory = "./data/uci_har/processed/train/split_data") {
        vector<string> files;
        if (!filesystem::exists(directory)) {
            if (rank == 0) {
                cout << "Directory does not exist: " << directory << endl;
            }
        } else {
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
                    file_to_data[filename].emplace_back(Point(features, -1, filename));
                }
            }
        }
        cout << "Worker " << rank << " loaded " << file_to_data[filename].size() 
             << " training points from " << filename << endl;
    }

    void loadTestData(const string& filename = "./data/uci_har/processed/test/X_test_pca.csv") {
        if (rank != 0) return;  // Only master loads test data
        
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Master could not open test file: " << filename << endl;
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
                if (features.size() == dimensions) {
                    test_data.emplace_back(Point(features, -1, filename));
                }
            }
        }
        cout << "Master loaded " << test_data.size() << " test points from " << filename << endl;
    }

    void distributeData() {
        if (rank == 0) {
            vector<string> train_files = findDataFiles();
            int num_workers = size - 1;

            cout << "Distributing " << train_files.size() << " training files to " << num_workers << " workers" << endl;

            // Distribute training files
            for (int i = 0; i < train_files.size(); i++) {
                int worker = (i % num_workers) + 1;
                string filename = train_files[i];
                int len = filename.length();

                MPI_Send(&len, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                MPI_Send(filename.c_str(), len, MPI_CHAR, worker, 1, MPI_COMM_WORLD);
            }

            // Send termination signal to all workers
            for (int worker = 1; worker < size; worker++) {
                int len = 0;
                MPI_Send(&len, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            }

            // Load test data globally on master
            loadTestData();

        } else {
            while (true) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (len == 0) break;

                char* buffer = new char[len + 1];
                MPI_Recv(buffer, len, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                buffer[len] = '\0';

                loadData(string(buffer));
                delete[] buffer;
            }
        }
    }

    void initialiseCentroids() {
        if (rank == 0) {
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
        } else {
            // Workers need to initialise their global_centroids structure
            global_centroids.resize(k);
            for (int i = 0; i < k; i++) {
                global_centroids[i].center.resize(dimensions);
            }
        }
        
        // Broadcast to all processes (including workers)
        broadcastCentroids();
    }

    void broadcastCentroids() {
        // Broadcast dimensions first
        MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Broadcast each centroid
        for (int i = 0; i < k; i++) {
            MPI_Bcast(global_centroids[i].center.data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
        local_centroids.assign(k, Centroid(dimensions));

        for (auto& [filename, data] : file_to_data) {
            for (auto& point : data) {
                double min_dist = numeric_limits<double>::max();
                int best_cluster = 0;

                for (int i = 0; i < k; i++) {
                    double dist = euclideanDistance(point.features, global_centroids[i].center);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = i;
                    }
                }

                point.label = best_cluster;
                local_centroids[best_cluster].count++;
                for (int j = 0; j < dimensions; j++) {
                    local_centroids[best_cluster].center[j] += point.features[j];
                }
            }
        }

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
            vector<Centroid> aggregated_centroids(k, Centroid(dimensions));
            vector<int> total_counts(k, 0);

            // Receive from all workers
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

            // Update global centroids
            for (int i = 0; i < k; i++) {
                if (total_counts[i] > 0) {
                    for (int j = 0; j < dimensions; j++) {
                        global_centroids[i].center[j] = aggregated_centroids[i].center[j] / total_counts[i];
                    }
                }
            }
        } else {
            // Workers send their local centroids
            for (int i = 0; i < k; i++) {
                MPI_Send(local_centroids[i].center.data(), dimensions,
                    MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
                MPI_Send(&local_centroids[i].count, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
            }
        }
        
        // Broadcast updated centroids to all processes
        broadcastCentroids();
    }

    double computeLocalInertia() {
        double inertia = 0.0;
        
        for (const auto& [filename, data] : file_to_data) {
            for (const auto& point : data) {
                if (point.label >= 0 && point.label < k) {
                    double dist = euclideanDistance(point.features, global_centroids[point.label].center);
                    inertia += dist * dist;
                }
            }
        }
        return inertia;
    }

    double computeGlobalInertia() {
        double local_inertia = computeLocalInertia();
        double global_inertia = 0.0;
        
        // All processes must participate in MPI_Reduce
        MPI_Reduce(&local_inertia, &global_inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // Broadcast the result back to all processes so they all have the same value
        MPI_Bcast(&global_inertia, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        return global_inertia;
    }

    void exportCentroids() {
        if (rank != 0) return;

        // Create output directory
        filesystem::create_directories("./fed_cluster_assignments");
        
        ofstream centroids_file("./fed_cluster_assignments/final_centroids.csv");
        if (!centroids_file.is_open()) {
            cout << "Could not create centroids file" << endl;
            return;
        }

        // Write header
        centroids_file << "cluster_id";
        for (int j = 0; j < dimensions; j++) {
            centroids_file << ",feature_" << j;
        }
        centroids_file << "\n";

        // Write centroid data
        for (int i = 0; i < k; i++) {
            centroids_file << i;
            for (int j = 0; j < dimensions; j++) {
                centroids_file << "," << fixed << setprecision(6) << global_centroids[i].center[j];
            }
            centroids_file << "\n";
        }
        centroids_file.close();
        cout << "Exported final centroids to ./fed_cluster_assignments/final_centroids.csv" << endl;
    }

    void exportTestAssignments() {
        if (rank != 0) return;  // Only master exports test assignments
        
        // Create output directory
        filesystem::create_directories("./fed_cluster_assignments");
        
        ofstream assignments_file("./fed_cluster_assignments/test_assignments.csv");
        
        if (!assignments_file.is_open()) {
            cout << "Could not create test assignments file" << endl;
            return;
        }

        // Write header
        assignments_file << "point_index,cluster_assignment";
        for (int j = 0; j < dimensions; j++) {
            assignments_file << ",feature_" << j;
        }
        assignments_file << "\n";

        // Write test point assignments
        for (int i = 0; i < test_data.size(); i++) {
            const auto& point = test_data[i];
            assignments_file << i << "," << point.label;
            for (int j = 0; j < dimensions; j++) {
                assignments_file << "," << fixed << setprecision(6) << point.features[j];
            }
            assignments_file << "\n";
        }
        
        assignments_file.close();
        cout << "Exported test assignments to ./fed_cluster_assignments/test_assignments.csv" << endl;
    }

    void train() {
        initialiseCentroids();
        
        double prev_inertia = numeric_limits<double>::max();
        bool converged = false;
        
        for (int iteration = 0; iteration < max_iterations; iteration++) {
            if (rank == 0) {
                cout << "Starting iteration " << (iteration + 1) << "..." << endl;
            }
            
            localKMeansStep();
            
            if (rank == 0) {
                cout << "Local K-means step complete, starting federated averaging..." << endl;
            }
            
            federatedAveraging();
            
            if (rank == 0) {
                cout << "Federated averaging complete, computing inertia..." << endl;
            }
            
            double current_inertia = computeGlobalInertia();
            
            // Check convergence on master and broadcast decision to all processes
            if (rank == 0) {
                cout << "Iteration " << iteration + 1 << ", Inertia: "
                    << fixed << setprecision(6) << current_inertia << endl;
                
                if (abs(prev_inertia - current_inertia) < tolerance) {
                    cout << "Converged after " << iteration + 1 << " iterations" << endl;
                    converged = true;
                }
                prev_inertia = current_inertia;
            }
            
            // Broadcast convergence decision to all processes
            int converged_flag = converged ? 1 : 0;
            MPI_Bcast(&converged_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            converged = (converged_flag == 1);
            
            // All processes must make the same decision about breaking
            if (converged) {
                if (rank == 0) {
                    cout << "All processes breaking from training loop." << endl;
                }
                break;
            }
        }
        
        if (rank == 0) {
            cout << "Training loop completed." << endl;
        }
    }

    void test() {
        if (rank == 0) {
            cout << "\n=== Testing Phase ===" << endl;
            cout << "Assigning test data to clusters using trained centroids..." << endl;

            // Assign cluster labels to test data using trained centroids
            for (auto& point : test_data) {
                double min_dist = numeric_limits<double>::max();
                int best_cluster = 0;

                for (int i = 0; i < k; i++) {
                    double dist = euclideanDistance(point.features, global_centroids[i].center);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = i;
                    }
                }
                point.label = best_cluster;
            }

            exportCentroids();
            exportTestAssignments();
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
    cout << "Process " << rank << " running on node " << hostname << endl;
    
    if (argc < 2) {
        if (rank == 0) {
            cerr << "Usage: " << argv[0] << " <k_clusters>" << endl;
            cerr << "Example: " << argv[0] << " 2" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int k = stoi(argv[1]);
    
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
    
    // Data distribution (both training and test data)
    fed_kmeans.distributeData();
    
    // Training
    double start_time = MPI_Wtime();
    fed_kmeans.train();
    double fed_time = MPI_Wtime() - start_time;
    
    // Testing
    double test_start_time = MPI_Wtime();
    fed_kmeans.test();
    double test_time = MPI_Wtime() - test_start_time;
    
    // Final results
    if (rank == 0) {
        cout << "\n=== Results ===" << endl;
        cout << "Federated training time: " << fed_time << " seconds" << endl;
        cout << "Testing time: " << test_time << " seconds" << endl;
        cout << "Total time: " << (fed_time + test_time) << " seconds" << endl;
        cout << "\nExported files:" << endl;
        cout << "- Final centroids: ./fed_cluster_assignments/final_centroids.csv" << endl;
        cout << "- Test assignments: ./fed_cluster_assignments/test_assignments.csv" << endl;
    }
    
    MPI_Finalize();
    return 0;
}