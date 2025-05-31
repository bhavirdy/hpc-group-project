#include <vector>
#include <sstream>
#include <random>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <bits/stdc++.h>

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

class CentralisedKMeans {
private:
    int k;
    int dimensions;
    int max_iterations;
    double tolerance;
    vector<Point> data;
    vector<Point> test_data;
    vector<Centroid> centroids;
    string export_dir;
  
public:
    CentralisedKMeans(int k_clusters, int max_iter = 100, double tol = 1e-6) 
                      : k(k_clusters), max_iterations(max_iter), tolerance(tol), 
                        export_dir("cent_cluster_assignments") {}
    
    void loadData(const string& filename = "./data/uci_har/processed/train/X_train_pca.csv") {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Could not open " << filename << endl;
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
            cout << "Centralised: Loaded " << data.size() << " points with " 
                     << dimensions << " dimensions" << endl;
        }
    }
    
    void loadTestData(const string& filename = "./data/uci_har/processed/test/X_test_pca.csv") {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Could not open test file " << filename << endl;
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
        
        cout << "Centralised: Loaded " << test_data.size() << " test points" << endl;
    }
    
    double euclideanDistance(const vector<double>& a, const vector<double>& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }
    
    void initializeCentroidsKMeansPlusPlus() {
        if (data.empty()) return;
        
        random_device rd;
        mt19937 gen(rd());
        
        centroids.resize(k);
        
        // Step 1: Choose first centroid randomly
        uniform_int_distribution<int> uniform_dist(0, data.size() - 1);
        int first_index = uniform_dist(gen);
        centroids[0] = Centroid(data[first_index].features);
        
        cout << "K-means++: Selected first centroid from point " << first_index << endl;
        
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
            cout << "K-means++: Selected centroid " << c + 1 << " from point " 
                << selected_index << " (distance weight: " << fixed << setprecision(4) 
                << distances[selected_index] << ")" << endl;
        }
        
        cout << "K-means++: Initialization complete with " << k << " centroids" << endl;
    }

    void train() {
        if (data.empty()) return;
        
        initializeCentroidsKMeansPlusPlus();
        
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
            
            cout << "Centralised Iteration " << iteration + 1 << ", Inertia: " 
                     << fixed << setprecision(6) << inertia << endl;
                        
            // Check for convergence
            if (abs(prev_inertia - inertia) < tolerance) {
                cout << "Centralised converged after " << iteration + 1 << " iterations" << endl;
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
        
        cout << "\n=== Testing on Test Data ===" << endl;
        
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
    
    void exportCentroids(const string& filename = "final_centroids.csv") {
        if (centroids.empty()) {
            cout << "No centroids to export" << endl;
            return;
        }
        
        // Create export directory if it doesn't exist
        filesystem::create_directories(export_dir);
        string filepath = export_dir + "/" + filename;
        
        ofstream file(filepath);
        if (!file.is_open()) {
            cout << "Could not create centroids file: " << filepath << endl;
            return;
        }
        
        // Write header
        file << "cluster_id";
        for (int j = 0; j < dimensions; j++) {
            file << ",feature_" << j;
        }
        file << "\n";
        
        // Write centroids
        for (int i = 0; i < k; i++) {
            file << i;
            for (int j = 0; j < dimensions; j++) {
                file << "," << fixed << setprecision(8) << centroids[i].center[j];
            }
            file << "\n";
        }
        
        file.close();
        cout << "Centroids exported to: " << filepath << endl;
    }
    
    void exportTestAssignments(const string& filename = "test_assignments.csv") {
        if (test_data.empty()) {
            cout << "No test data to export" << endl;
            return;
        }
        
        // Create export directory if it doesn't exist
        filesystem::create_directories(export_dir);
        string filepath = export_dir + "/" + filename;
        
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
    
    void exportAll() {
        cout << "\n=== Exporting Results ===" << endl;
        exportCentroids();
        if (!test_data.empty()) {
            exportTestAssignments();
        }
    }
};

int main(int argc, char* argv[]) {
    
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <k_clusters>" << endl;
        cerr << "Example: " << argv[0] << " 2" << endl;
        return 1;
    }
    
    int k = stoi(argv[1]);
    
    cout << "\n=== Centralised K-Means (Baseline) ===" << endl;
    CentralisedKMeans cent_kmeans(k);
    
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
    
    // Export all results
    cent_kmeans.exportAll();

    return 0;
}