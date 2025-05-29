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
    int k;
    int dimensions = 100;
    int max_iterations;
    double tolerance;
    map<string, vector<Point>> file_to_data;
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
                    file_to_data[filename].emplace_back(Point(features));
                }
            }
        }
        cout << "Worker " << rank << " loaded " << file_to_data[filename].size() << " points from " << filename << endl;
    }

    void distributeData() {
        if (rank == 0) {
            vector<string> files = findDataFiles();
            int num_workers = size - 1;

            cout << "Distributing " << files.size() << " files to " << num_workers << " workers" << endl;

            for (int i = 0; i < files.size(); i++) {
                int worker = (i % num_workers) + 1;
                string filename = files[i];
                int len = filename.length();

                MPI_Send(&len, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                MPI_Send(filename.c_str(), len, MPI_CHAR, worker, 1, MPI_COMM_WORLD);
            }

            for (int worker = 1; worker < size; worker++) {
                int len = 0;
                MPI_Send(&len, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            }

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

            broadcastCentroids();
        } else {
            receiveCentroids();
        }
    }

    void broadcastCentroids() {
        for (int worker = 1; worker < size; worker++) {
            MPI_Send(&dimensions, 1, MPI_INT, worker, 16, MPI_COMM_WORLD);
        }

        for (int worker = 1; worker < size; worker++) {
            for (int i = 0; i < k; i++) {
                MPI_Send(global_centroids[i].center.data(), dimensions,
                    MPI_DOUBLE, worker, 3, MPI_COMM_WORLD);
            }
        }
    }

    void receiveCentroids() {
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

            for (int i = 0; i < k; i++) {
                if (total_counts[i] > 0) {
                    for (int j = 0; j < dimensions; j++) {
                        global_centroids[i].center[j] = aggregated_centroids[i].center[j] / total_counts[i];
                    }
                }
            }

            broadcastCentroids();
        } else {
            for (int i = 0; i < k; i++) {
                MPI_Send(local_centroids[i].center.data(), dimensions,
                    MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
                MPI_Send(&local_centroids[i].count, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
            }

            receiveCentroids();
        }
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
        MPI_Reduce(&local_inertia, &global_inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        return global_inertia;
    }

    void exportClusterAssignments(const string& output_dir = "./cluster_assignments") {
        if (!filesystem::exists(output_dir)) {
            filesystem::create_directories(output_dir);
        }

        if (file_to_data.empty()) {
                cout << "Worker " << rank << " has no data to export." << endl;
                return;
            }

        for (const auto& [filepath, data] : file_to_data) {
            string filename = filesystem::path(filepath).stem();
            string out_path = output_dir + "/worker_" + to_string(rank) + "_" + filename + "_assignments.csv";

            ofstream out(out_path);
            if (!out.is_open()) {
                cerr << "Worker " << rank << " failed to open " << out_path << " for writing." << endl;
                continue;
            }

            for (const auto& point : data) {
                for (size_t i = 0; i < point.features.size(); ++i) {
                    out << point.features[i];
                    if (i != point.features.size() - 1) {
                        out << ",";
                    }
                }
                out << "," << point.label << "\n";
            }

            out.close();
            cout << "Worker " << rank << " wrote assignments for " << filepath << " to " << out_path << endl;
        }
    }

    void train() {
        initialiseCentroids();

        double prev_inertia = numeric_limits<double>::max();

        for (int iteration = 0; iteration < max_iterations; iteration++) {
            localKMeansStep();
            federatedAveraging();

            double current_inertia = computeGlobalInertia();

            if (rank == 0) {
                cout << "Iteration " << iteration + 1 << ", Inertia: "
                     << fixed << setprecision(6) << current_inertia << endl;

                if (abs(prev_inertia - current_inertia) < tolerance) {
                    cout << "Converged after " << iteration + 1 << " iterations" << endl;
                    break;
                }
            }

            prev_inertia = current_inertia;
        }

        exportClusterAssignments();
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
            
    // Centralized comparison
    if (rank == 0) {
        cout << "\n=== Centralized K-Means ===" << endl;
        CentralizedKMeans cent_kmeans(k);
        cent_kmeans.loadData(centralized_file);
        
        start_time = MPI_Wtime();
        cent_kmeans.train();
        double cent_time = MPI_Wtime() - start_time;
        
        cout << "Centralized training time: " << cent_time << " seconds" << endl;

        cout << "\n=== Performance Comparison ===" << endl;
        cout << "Federated time: " << fed_time << "s" << endl;
        cout << "Centralized time: " << cent_time << "s" << endl;
    }
    
    MPI_Finalize();
    return 0;
}