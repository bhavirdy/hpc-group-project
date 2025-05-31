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
#include <filesystem>
#include <stdexcept>

using namespace std;

struct Point {
    vector<double> features;
    int cluster_label;
    
    Point() : cluster_label(-1) {}
    Point(const vector<double>& f) : features(f), cluster_label(-1) {}
    Point(const vector<double>& f, int label) : features(f), cluster_label(label) {}
};

struct Centroid {
    vector<double> center;
    int point_count;
    
    Centroid() : point_count(0) {}
    Centroid(int dimensions) : center(dimensions, 0.0), point_count(0) {}
    Centroid(const vector<double>& c) : center(c), point_count(0) {}
    
    void reset(int dimensions) {
        center.assign(dimensions, 0.0);
        point_count = 0;
    }
};

class FederatedKMeans {
private:
    // MPI variables
    int rank, size;
    
    // Algorithm parameters
    int k_clusters;
    int dimensions;
    int max_iterations;
    double convergence_tolerance;
    
    // Data storage
    vector<Point> local_data;
    vector<Point> test_data;  // Only used by master
    
    // Clustering state
    vector<Centroid> global_centroids;
    vector<Centroid> local_centroids;

public:
    FederatedKMeans(int k, int max_iter = 100, double tolerance = 1e-6) 
        : k_clusters(k), dimensions(0), max_iterations(max_iter), convergence_tolerance(tolerance) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        if (size < 2) {
            throw runtime_error("Requires at least 2 MPI processes (1 master + 1 worker)");
        }
    }

    vector<string> findDataFiles(const string& directory = "./data/uci_har/processed/train/split_data") {
        vector<string> files;
        
        if (!filesystem::exists(directory)) {
            if (rank == 0) {
                cout << "Warning: Directory does not exist: " << directory << endl;
            }
            return files;
        }
        
        try {
            for (const auto& entry : filesystem::directory_iterator(directory)) {
                if (entry.path().extension() == ".csv") {
                    files.push_back(entry.path().string());
                }
            }
            sort(files.begin(), files.end());
        } catch (const exception& e) {
            if (rank == 0) {
                cout << "Error reading directory: " << e.what() << endl;
            }
        }
        
        return files;
    }

    bool loadDataFromFile(const string& filename, vector<Point>& data_container) {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Process " << rank << " failed to open: " << filename << endl;
            return false;
        }

        string line;
        bool first_line = true;
        int loaded_points = 0;

        while (getline(file, line)) {
            if (line.empty()) continue;
            
            // Skip header line
            if (first_line) {
                first_line = false;
                continue;
            }

            istringstream iss(line);
            vector<double> features;
            string value;

            // Parse CSV values
            while (getline(iss, value, ',')) {
                try {
                    features.push_back(stod(value));
                } catch (const exception&) {
                    // Skip invalid values
                    continue;
                }
            }

            if (!features.empty()) {
                // Set dimensions from first valid point
                if (dimensions == 0) {
                    dimensions = features.size();
                }
                
                // Only accept points with correct dimensions
                if (features.size() == dimensions) {
                    data_container.emplace_back(Point(features));
                    loaded_points++;
                }
            }
        }

        cout << "Process " << rank << " loaded " << loaded_points 
             << " points from " << filename << endl;
        return loaded_points > 0;
    }

    void distributeTrainingData() {
        if (rank == 0) {
            // Master distributes files to workers
            vector<string> train_files = findDataFiles();
            int num_workers = size - 1;

            if (train_files.empty()) {
                cout << "No training files found!" << endl;
                return;
            }

            cout << "Distributing " << train_files.size() 
                 << " files to " << num_workers << " workers" << endl;

            // Round-robin distribution of files to workers
            for (size_t i = 0; i < train_files.size(); i++) {
                int target_worker = (i % num_workers) + 1;
                string filename = train_files[i];
                int filename_length = filename.length();

                MPI_Send(&filename_length, 1, MPI_INT, target_worker, 0, MPI_COMM_WORLD);
                MPI_Send(filename.c_str(), filename_length, MPI_CHAR, target_worker, 1, MPI_COMM_WORLD);
            }

            // Send termination signals
            for (int worker = 1; worker < size; worker++) {
                int termination_signal = 0;
                MPI_Send(&termination_signal, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            }

            // Load test data on master
            loadTestData();

        } else {
            // Workers receive and load files
            while (true) {
                int filename_length;
                MPI_Recv(&filename_length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (filename_length == 0) break;  // Termination signal

                vector<char> filename_buffer(filename_length + 1);
                MPI_Recv(filename_buffer.data(), filename_length, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                filename_buffer[filename_length] = '\0';

                string filename(filename_buffer.data());
                loadDataFromFile(filename, local_data);
            }
        }

        // Synchronize dimensions across all processes
        MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    void loadTestData(const string& test_file = "./data/uci_har/processed/test/X_test_pca.csv") {
        if (rank != 0) return;  // Only master loads test data
        
        cout << "Loading test data..." << endl;
        if (loadDataFromFile(test_file, test_data)) {
            cout << "Loaded " << test_data.size() << " test points" << endl;
        }
    }

    void initializeCentroidsKMeansPlusPlus(const string& full_training_file = "./data/uci_har/processed/train/X_train_pca.csv") {
        global_centroids.resize(k_clusters);
        
        if (rank == 0) {
            // Master performs K-means++ initialization
            
            // Load all training data from the single file
            vector<Point> all_training_data;
            if (!full_training_file.empty() && loadDataFromFile(full_training_file, all_training_data)) {
            } else {
                return;
            }
                        
            random_device rd;
            mt19937 gen(rd());
            
            // Step 1: Choose first centroid randomly
            uniform_int_distribution<int> point_dist(0, all_training_data.size() - 1);
            int first_centroid_idx = point_dist(gen);
            
            global_centroids[0] = Centroid(dimensions);
            for (int j = 0; j < dimensions; j++) {
                global_centroids[0].center[j] = all_training_data[first_centroid_idx].features[j];
            }
                        
            // Step 2: Choose remaining centroids using K-means++ algorithm
            for (int c = 1; c < k_clusters; c++) {
                vector<double> distances(all_training_data.size());
                double total_distance = 0.0;
                
                // Calculate squared distance from each point to nearest existing centroid
                for (size_t i = 0; i < all_training_data.size(); i++) {
                    double min_dist_sq = numeric_limits<double>::max();
                    
                    // Find distance to nearest existing centroid
                    for (int existing_c = 0; existing_c < c; existing_c++) {
                        double dist = calculateDistance(all_training_data[i].features, 
                                                    global_centroids[existing_c].center);
                        double dist_sq = dist * dist;
                        if (dist_sq < min_dist_sq) {
                            min_dist_sq = dist_sq;
                        }
                    }
                    
                    distances[i] = min_dist_sq;
                    total_distance += min_dist_sq;
                }
                
                // Choose next centroid with probability proportional to squared distance
                uniform_real_distribution<double> prob_dist(0.0, total_distance);
                double target_distance = prob_dist(gen);
                
                double cumulative_distance = 0.0;
                int selected_idx = 0;
                
                for (size_t i = 0; i < all_training_data.size(); i++) {
                    cumulative_distance += distances[i];
                    if (cumulative_distance >= target_distance) {
                        selected_idx = i;
                        break;
                    }
                }
                
                // Set the new centroid
                global_centroids[c] = Centroid(dimensions);
                for (int j = 0; j < dimensions; j++) {
                    global_centroids[c].center[j] = all_training_data[selected_idx].features[j];
                }
            }
            
            
        } else {
            // Workers initialize empty centroid structures
            for (int i = 0; i < k_clusters; i++) {
                global_centroids[i] = Centroid(dimensions);
            }
        }

        // Broadcast initialized centroids to all processes
        broadcastCentroids();
        
        // Initialize local centroids
        local_centroids.resize(k_clusters);
        for (int i = 0; i < k_clusters; i++) {
            local_centroids[i] = Centroid(dimensions);
        }
    }

    void broadcastCentroids() {
        // Broadcast centroids from master to all processes
        for (int i = 0; i < k_clusters; i++) {
            MPI_Bcast(global_centroids[i].center.data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    double calculateDistance(const vector<double>& point1, const vector<double>& point2) {
        double sum_squared_diff = 0.0;
        for (size_t i = 0; i < point1.size(); i++) {
            double diff = point1[i] - point2[i];
            sum_squared_diff += diff * diff;
        }
        return sqrt(sum_squared_diff);
    }

    void performLocalClustering() {
        // Reset local centroids
        for (int i = 0; i < k_clusters; i++) {
            local_centroids[i].reset(dimensions);
        }

        // Assign points to nearest centroid and accumulate for new centroid calculation
        for (auto& point : local_data) {
            double min_distance = numeric_limits<double>::max();
            int best_cluster = 0;

            // Find nearest centroid
            for (int i = 0; i < k_clusters; i++) {
                double distance = calculateDistance(point.features, global_centroids[i].center);
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = i;
                }
            }

            // Assign point to cluster
            point.cluster_label = best_cluster;

            // Accumulate for centroid calculation
            local_centroids[best_cluster].point_count++;
            for (int j = 0; j < dimensions; j++) {
                local_centroids[best_cluster].center[j] += point.features[j];
            }
        }

        // Calculate local centroid averages
        for (int i = 0; i < k_clusters; i++) {
            if (local_centroids[i].point_count > 0) {
                for (int j = 0; j < dimensions; j++) {
                    local_centroids[i].center[j] /= local_centroids[i].point_count;
                }
            }
        }
    }

    void aggregateCentroids() {
        if (rank == 0) {
            // Master aggregates centroids from all workers
            vector<Centroid> aggregated_centroids(k_clusters, Centroid(dimensions));
            vector<int> total_point_counts(k_clusters, 0);

            // Collect centroids from all workers
            for (int worker = 1; worker < size; worker++) {
                for (int i = 0; i < k_clusters; i++) {
                    vector<double> worker_centroid(dimensions);
                    int worker_count;

                    MPI_Recv(worker_centroid.data(), dimensions, MPI_DOUBLE, worker, 100 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&worker_count, 1, MPI_INT, worker, 200 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    total_point_counts[i] += worker_count;
                    if (worker_count > 0) {
                        for (int j = 0; j < dimensions; j++) {
                            aggregated_centroids[i].center[j] += worker_centroid[j] * worker_count;
                        }
                    }
                }
            }

            // Update global centroids with weighted average
            for (int i = 0; i < k_clusters; i++) {
                if (total_point_counts[i] > 0) {
                    for (int j = 0; j < dimensions; j++) {
                        global_centroids[i].center[j] = aggregated_centroids[i].center[j] / total_point_counts[i];
                    }
                }
            }

        } else {
            // Workers send their local centroids to master
            for (int i = 0; i < k_clusters; i++) {
                MPI_Send(local_centroids[i].center.data(), dimensions, MPI_DOUBLE, 0, 100 + i, MPI_COMM_WORLD);
                MPI_Send(&local_centroids[i].point_count, 1, MPI_INT, 0, 200 + i, MPI_COMM_WORLD);
            }
        }

        // Broadcast updated centroids to all processes
        broadcastCentroids();
    }

    double calculateLocalInertia() {
        double inertia = 0.0;
        for (const auto& point : local_data) {
            if (point.cluster_label >= 0 && point.cluster_label < k_clusters) {
                double distance = calculateDistance(point.features, global_centroids[point.cluster_label].center);
                inertia += distance * distance;
            }
        }
        return inertia;
    }

    double calculateGlobalInertia() {
        double local_inertia = calculateLocalInertia();
        double global_inertia = 0.0;
        
        MPI_Reduce(&local_inertia, &global_inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_inertia, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        return global_inertia;
    }

    void train() {
        if (rank > 0) {
            cout << "Process " << rank << " starting training with " << local_data.size() << " data points" << endl;
        }
        
        initializeCentroidsKMeansPlusPlus();
        
        double previous_inertia = numeric_limits<double>::max();
        bool has_converged = false;
        
        for (int iteration = 0; iteration < max_iterations; iteration++) {
            // Local K-means step
            performLocalClustering();
            
            // Federated averaging
            aggregateCentroids();
            
            // Check convergence
            double current_inertia = calculateGlobalInertia();
            
            if (rank == 0) {
                
                if (abs(previous_inertia - current_inertia) < convergence_tolerance) {
                    cout << "Converged after " << (iteration + 1) << " iterations" << endl;
                    has_converged = true;
                }
                previous_inertia = current_inertia;
            }
            
            // Broadcast convergence decision
            int convergence_flag = has_converged ? 1 : 0;
            MPI_Bcast(&convergence_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            has_converged = (convergence_flag == 1);
            
            if (has_converged) break;
        }
        
        if (rank == 0) {
            cout << "Training completed" << endl;
        }
    }

    void runInference() {
        if (rank != 0) return;  // Only master runs inference

        cout << "\n=== Running Inference on Test Data ===" << endl;
        
        // Assign cluster labels to test data
        for (auto& point : test_data) {
            double min_distance = numeric_limits<double>::max();
            int best_cluster = 0;

            for (int i = 0; i < k_clusters; i++) {
                double distance = calculateDistance(point.features, global_centroids[i].center);
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = i;
                }
            }
            point.cluster_label = best_cluster;
        }

        cout << "Assigned " << test_data.size() << " test points to clusters" << endl;

        // Display cluster distribution for test data
        vector<int> cluster_counts(k_clusters, 0);
        for (const auto& point : test_data) {
            cluster_counts[point.cluster_label]++;
        }
        cout << "Test data cluster distribution:" << endl;
        for (int i = 0; i < k_clusters; i++) {
            cout << "Cluster " << i << ": " << cluster_counts[i] << " points" << endl;
        }
    }

    void exportResults() {
        if (rank != 0) return;  // Only master exports

        try {
            filesystem::create_directories("./fed_cluster_results");
            
            // Export centroids
            exportCentroids();
            
            // Export test assignments
            exportTestAssignments();
            
        } catch (const exception& e) {
            cout << "Error creating output directory: " << e.what() << endl;
        }
    }

private:
    void exportCentroids() {
        ofstream file("./fed_cluster_results/centroids.csv");
        if (!file.is_open()) {
            cout << "Failed to create centroids file" << endl;
            return;
        }

        // Write header
        file << "cluster_id";
        for (int j = 0; j < dimensions; j++) {
            file << ",feature_" << j;
        }
        file << "\n";

        // Write centroid data
        for (int i = 0; i < k_clusters; i++) {
            file << i;
            for (int j = 0; j < dimensions; j++) {
                file << "," << fixed << setprecision(6) << global_centroids[i].center[j];
            }
            file << "\n";
        }
        
        cout << "\nExported centroids to ./fed_cluster_results/centroids.csv" << endl;
    }

    void exportTestAssignments() {
        ofstream file("./fed_cluster_assignments/test_assignments.csv");
        if (!file.is_open()) {
            cout << "Failed to create test assignments file" << endl;
            return;
        }

        // Write header
        file << "point_index,cluster_assignment";
        for (int j = 0; j < dimensions; j++) {
            file << ",feature_" << j;
        }
        file << "\n";

        // Write test point assignments
        for (size_t i = 0; i < test_data.size(); i++) {
            const auto& point = test_data[i];
            file << i << "," << point.cluster_label;
            for (int j = 0; j < dimensions; j++) {
                file << "," << fixed << setprecision(6) << point.features[j];
            }
            file << "\n";
        }
        
        cout << "Exported test assignments to ./fed_cluster_results/test_assignments.csv" << endl;
    }
};

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        // Parse command line arguments
        if (argc < 2) {
            if (rank == 0) {
                cout << "Usage: " << argv[0] << " <k_clusters>" << endl;
                cout << "Example: " << argv[0] << " 6 " << endl;
            }
            MPI_Finalize();
            return 1;
        }
        
        int k = stoi(argv[1]);
        int max_iter = (argc > 2) ? stoi(argv[2]) : 100;
        double tolerance = (argc > 3) ? stod(argv[3]) : 1e-6;
        
        // Create and run federated K-means
        FederatedKMeans fed_kmeans(k, max_iter, tolerance);
        
        // Data distribution phase
        double start_time = MPI_Wtime();
        fed_kmeans.distributeTrainingData();
        double distribution_time = MPI_Wtime() - start_time;
        
        // Training phase
        double training_start = MPI_Wtime();
        fed_kmeans.train();
        double training_time = MPI_Wtime() - training_start;
        
        // Inference phase
        double inference_start = MPI_Wtime();
        fed_kmeans.runInference();
        double inference_time = MPI_Wtime() - inference_start;
        
        // Export results
        fed_kmeans.exportResults();
        
        double total_time = MPI_Wtime() - start_time;
        
        // Display results
        if (rank == 0) {
            cout << "\n=== Performance Results ===" << endl;
            cout << "Data distribution time: " << fixed << setprecision(3) << distribution_time << " seconds" << endl;
            cout << "Training time: " << training_time << " seconds" << endl;
            cout << "Inference time: " << inference_time << " seconds" << endl;
            cout << "Total execution time: " << total_time << " seconds" << endl;
            cout << "\nOutput files:" << endl;
            cout << "- Centroids: ./fed_cluster_results/centroids.csv" << endl;
            cout << "- Test assignments: ./fed_cluster_results/test_assignments.csv" << endl;
        }
        
    } catch (const exception& e) {
        cout << "Process " << rank << " encountered error: " << e.what() << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}