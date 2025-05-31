#include <vector>
#include <sstream>
#include <random>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>

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
    vector<Centroid> centroids;
    string export_dir;
    
    void createExportDirectory() {
        try {
            filesystem::create_directories(export_dir);
            cout << "Created export directory: " << export_dir << endl;
        } catch (const filesystem::filesystem_error& e) {
            cerr << "Error creating directory " << export_dir << ": " << e.what() << endl;
        }
    }
    
    void exportClusterAssignments(int iteration) {
        string filename = export_dir + "/assignments_round_" + to_string(iteration + 1) + ".csv";
        ofstream file(filename);
        
        if (!file.is_open()) {
            cerr << "Could not create assignments file: " << filename << endl;
            return;
        }
        
        // Write header
        file << "point_id,cluster_id";
        for (int i = 0; i < dimensions; i++) {
            file << ",feature_" << i;
        }
        file << endl;
        
        // Write data
        for (size_t i = 0; i < data.size(); i++) {
            file << i << "," << data[i].label;
            for (int j = 0; j < dimensions; j++) {
                file << "," << fixed << setprecision(8) << data[i].features[j];
            }
            file << endl;
        }
        
        file.close();
        cout << "Exported cluster assignments to: " << filename << endl;
    }
    
    void exportCentroids(int iteration) {
        string filename = export_dir + "/centroids_round_" + to_string(iteration + 1) + ".csv";
        ofstream file(filename);
        
        if (!file.is_open()) {
            cerr << "Could not create centroids file: " << filename << endl;
            return;
        }
        
        // Write header
        file << "centroid_id,count";
        for (int i = 0; i < dimensions; i++) {
            file << ",center_" << i;
        }
        file << endl;
        
        // Write centroids
        for (int i = 0; i < k; i++) {
            file << i << "," << centroids[i].count;
            for (int j = 0; j < dimensions; j++) {
                file << "," << fixed << setprecision(8) << centroids[i].center[j];
            }
            file << endl;
        }
        
        file.close();
        cout << "Exported centroids to: " << filename << endl;
    }
       
public:
    CentralisedKMeans(int k_clusters, int max_iter = 100, double tol = 1e-6) 
                      : k(k_clusters), max_iterations(max_iter), tolerance(tol), 
                        export_dir("cent_cluster_assignments") {}
    
    void loadData(const string& filename = "./data/uci_har/processed_data/split_data") {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Could not open " << filename << " for centralised comparison" << endl;
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
            cout << "Centralised: Loaded " << data.size() << " points with " 
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
        
        // Create export directory
        createExportDirectory();
        
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
        bool converged = false;
        int final_iteration = 0;
        
        for (int iteration = 0; iteration < max_iterations; iteration++) {
            final_iteration = iteration;
            
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
        
        // Export final results only
        exportClusterAssignments(final_iteration);
        exportCentroids(final_iteration);
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
    cent_kmeans.loadData();
    
    auto start = chrono::high_resolution_clock::now();
    cent_kmeans.train();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    cout << "Centralised training time: " << elapsed.count() << " seconds" << endl;

    return 0;
}