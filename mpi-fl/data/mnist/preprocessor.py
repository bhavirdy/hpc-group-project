const int image_width = 28;
const int image_height = 28;
const double min_value = 0.0;
const double max_value = 255.0;    

// Normalize pixel values from 0-255 to 0-1 range
    void normalizeFeatures(vector<double>& features) {
        for (double& f : features) {
            f = (f - min_value) / (max_value - min_value);
            // Clamp to [0, 1] range
            f = max(0.0, min(1.0, f));
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