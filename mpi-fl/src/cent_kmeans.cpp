#include <vector>
#include <sstream>
#include <random>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

using namespace std;

// Configuration struct to hold all file paths
struct DatasetConfig {
    string name;
    string train_file;
    string test_file;
    string export_dir;
    
    DatasetConfig(const string& dataset_name, 
                  const string& train_path, 
                  const string& test_path, 
                  const string& export_path) 
        : name(dataset_name), train_file(train_path), test_file(test_path), export_dir(export_path) {}
};

// Dataset configurations
class DatasetConfigurations {
public:
    static DatasetConfig getUCIHAR() {
        return DatasetConfig(
            "UCI-HAR",
            "./data/uci_har/processed/train/X_train_pca.csv",
            "./data/uci_har/processed/test/X_test_pca.csv",
            "./results/uci_har/cent_cluster_assignments"
        );
    }
    
    static DatasetConfig getMNIST() {
        return DatasetConfig(
            "MNIST",
            "./data/mnist/processed/train/X_train_pca.csv",
            "./data/mnist/processed/test/X_test_pca.csv",
            "./results/mnist/cent_cluster_assignments"
        );
    }
    
    static vector<string> getAvailableDatasets() {
        return {"uci-har", "mnist"};
    }
    
    static DatasetConfig getConfig(const string& dataset_name) {
        if (dataset_name == "uci-har") {
            return getUCIHAR();
        } else if (dataset_name == "mnist") {
            return getMNIST();
        } else {
            throw invalid_argument("Unknown dataset: " + dataset_name);
        }
    }
};

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

class CentralisedKMeans {
private:
    int k;
    int dimensions;
    int max_iterations;
    double tolerance;
    vector<Point> data;
    vector<Point> test_data;
    vector<Centroid> centroids;
    DatasetConfig config;
  
public:
    CentralisedKMeans(int k_clusters, const DatasetConfig& dataset_config, int max_iter = 100, double tol = 1e-6) 
                      : k(k_clusters), config(dataset_config), max_iterations(max_iter), tolerance(tol) {}
    
    void loadData() {
        ifstream file(config.train_file);
        if (!file.is_open()) {
            cout << "Could not open " << config.train_file << endl;
            return;
        }
        
        string line;
        getline(file, line);
        
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
            cout << "Loaded " << data.size() << " training points from " << config.name << " dataset" << endl; 
        }
    }
    
    void loadTestData() {
        ifstream file(config.test_file);
        if (!file.is_open()) {
            cout << "Could not open test file " << config.test_file << endl;
            return;
        }
        
        string line;
        getline(file, line);
        
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
                test_data.push_back(Point(features));
            }
        }
        
        cout << "Loaded " << test_data.size() << " test points from " << config.name << " dataset" << endl;
    }
    
    double euclideanDistance(const vector<double>& a, const vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    
    void initialiseCentroidsKMeansPlusPlus() {
        if (data.empty()) return;
        
        random_device rd;
        mt19937 gen(rd());
        
        centroids.resize(k);
        
        // Step 1: Choose first centroid randomly
        uniform_int_distribution<int> uniform_dist(0, data.size() - 1);
        int first_index = uniform_dist(gen);
        centroids[0] = Centroid(data[first_index].features);
                
        // Step 2: Choose remaining centroids using K-means++ method
        for (int c = 1; c < k; c++) {
            vector<double> distances(data.size());
            double total_distance = 0.0;
            
            // Calculate squared distance from each point to nearest existing centroid
            for (int i = 0; i < data.size(); i++) {
                double min_dist_sq = numeric_limits<double>::max();
                
                // Find distance to nearest centroid
                for (int j = 0; j < c; j++) {
                    double dist = euclideanDistance(data[i].features, centroids[j].center);
                    double dist_sq = dist * dist;
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                    }
                }
                
                distances[i] = min_dist_sq;
                total_distance += min_dist_sq;
            }
            
            // Choose next centroid with probability proportional to squared distance
            uniform_real_distribution<double> real_dist(0.0, total_distance);
            double random_value = real_dist(gen);
            
            double cumulative_distance = 0.0;
            int selected_index = 0;
            
            for (int i = 0; i < data.size(); i++) {
                cumulative_distance += distances[i];
                if (cumulative_distance >= random_value) {
                    selected_index = i;
                    break;
                }
            }
            
            centroids[c] = Centroid(data[selected_index].features);
        }
    }

    void train() {
        if (data.empty()) return;
        
        initialiseCentroidsKMeansPlusPlus();
        
        double prev_inertia = numeric_limits<double>::max();
        bool converged = false;
        
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
                        
            // Check for convergence
            if (abs(prev_inertia - inertia) < tolerance) {
                cout << "Centralised K-Means converged after " << iteration + 1 << " iterations" << endl;
                converged = true;
                break;
            }
            
            prev_inertia = inertia;
        }
    }
    
    void test() {
        if (test_data.empty() || centroids.empty()) {
            cout << "No test data or centroids available for testing" << endl;
            return;
        }
        
        cout << "\n=== Testing on " << config.name << " Test Data ===" << endl;
        
        // Assign test points to clusters
        for (auto& point : test_data) {
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
        
        // Compute test inertia
        double test_inertia = 0.0;
        for (const auto& point : test_data) {
            double dist = euclideanDistance(point.features, centroids[point.label].center);
            test_inertia += dist * dist;
        }
        
        cout << "Test inertia: " << fixed << setprecision(6) << test_inertia << endl;
        
        // Display cluster distribution for test data
        vector<int> cluster_counts(k, 0);
        for (const auto& point : test_data) {
            cluster_counts[point.label]++;
        }
        
        cout << "Test data cluster distribution:" << endl;
        for (int i = 0; i < k; i++) {
            cout << "Cluster " << i << ": " << cluster_counts[i] << " points" << endl;
        }
    }
       
    void exportTestAssignments(const string& filename = "test_assignments.csv") {
        if (test_data.empty()) {
            cout << "No test data to export" << endl;
            return;
        }
        
        // Create export directory if it doesn't exist
        filesystem::create_directories(config.export_dir);
        string filepath = config.export_dir + "/" + filename;
        
        ofstream file(filepath);
        if (!file.is_open()) {
            cout << "Could not create test assignments file: " << filepath << endl;
            return;
        }
        
        // Write header
        file << "point_index,cluster_assignment";
        for (int j = 0; j < dimensions; j++) {
            file << ",feature_" << j;
        }
        file << "\n";

        // Write test point assignments
        for (int i = 0; i < test_data.size(); i++) {
            const auto& point = test_data[i];
            file << i << "," << point.label;
            for (int j = 0; j < dimensions; j++) {
                file << "," << fixed << setprecision(6) << point.features[j];
            }
            file << "\n";
        }
        
        file.close();
        cout << "Test assignments exported to: " << filepath << endl;
    }
};

void printUsage(const string& program_name) {
    cout << "Usage: " << program_name << " <k_clusters> <dataset>" << endl;
    cout << "Parameters:" << endl;
    cout << "  k_clusters: Number of clusters (e.g., 2, 5, 10)" << endl;
    cout << "  dataset:    Dataset to use" << endl;
    
    auto available_datasets = DatasetConfigurations::getAvailableDatasets();
    cout << "Available datasets: ";
    for (size_t i = 0; i < available_datasets.size(); i++) {
        cout << available_datasets[i];
        if (i < available_datasets.size() - 1) cout << ", ";
    }
    cout << endl;
    
    cout << "\nExamples:" << endl;
    cout << "  " << program_name << " 6 uci-har" << endl;
    cout << "  " << program_name << " 10 mnist" << endl;
}

int main(int argc, char* argv[]) {
    
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    int k = stoi(argv[1]);
    string dataset_name = argv[2];
    
    // Validate dataset name
    auto available_datasets = DatasetConfigurations::getAvailableDatasets();
    bool valid_dataset = false;
    for (const auto& ds : available_datasets) {
        if (ds == dataset_name) {
            valid_dataset = true;
            break;
        }
    }
    
    if (!valid_dataset) {
        cout << "Error: Invalid dataset '" << dataset_name << "'" << endl;
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        // Get dataset configuration
        DatasetConfig config = DatasetConfigurations::getConfig(dataset_name);
        
        cout << "\n=== Centralised K-Means (Baseline) ===" << endl;
        cout << "Dataset: " << config.name << endl;
        cout << "K clusters: " << k << endl;
        
        CentralisedKMeans cent_kmeans(k, config);
        
        // Load training data
        cent_kmeans.loadData();
        
        // Load test data
        cent_kmeans.loadTestData();
        
        // Train the model
        auto start = chrono::high_resolution_clock::now();
        cent_kmeans.train();
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        
        cout << "Centralised training time: " << elapsed.count() << " seconds" << endl;
        
        // Test the model
        cent_kmeans.test();
        
        // Export results
        cent_kmeans.exportTestAssignments();
        
    } catch (const exception& e) {
        cout << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}