#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

bool readPGM(const string &filename, vector<vector<int>> &image, int &width, int &height, int &maxVal) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Unable to open input file." << endl;
        return false;
    }
    
    string format;
    file >> format;
    if (format != "P2") {
        cerr << "Unsupported PGM format!" << endl;
        return false;
    }
    
    file >> width >> height >> maxVal;
    image.resize(height, vector<int>(width));
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            file >> image[i][j];
        }
    }
    
    file.close();
    return true;
}

void writePGM(const string &filename, const vector<vector<int>> &image, int width, int height, int maxVal) {
    ofstream file(filename, ios::binary);
    file << "P2\n" << width << " " << height << "\n" << maxVal << "\n";
    
    for (const auto &row : image) {
        for (int pixel : row) {
            file << pixel << " ";
        }
        file << "\n";
    }
    
    file.close();
}

// Function to apply Gaussian filter
vector<vector<int>> applyGaussianFilter(const vector<vector<int>> &image, int width, int height, double sigma_x, double sigma_y, int kernelSize) {
    int halfSize = kernelSize / 2;
    vector<vector<double>> kernel(kernelSize, vector<double>(kernelSize));
    double sum = 0.0;
    
// Compute Gaussian Kernel
    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            double value = exp(-(((i * i) / (2 * sigma_x * sigma_x)) + ((j * j) / (2 * sigma_y * sigma_y))));
            kernel[i + halfSize][j + halfSize] = value;
            sum += value;
        }
    }

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    vector<vector<int>> output(height, vector<int>(width));

// Apply Gaussian filter
    for (int y = halfSize; y < height - halfSize; y++) {
        for (int x = halfSize; x < width - halfSize; x++) {
            double sum = 0.0;
            for (int i = -halfSize; i <= halfSize; i++) {
                for (int j = -halfSize; j <= halfSize; j++) {
                    sum += image[y + i][x + j] * kernel[i + halfSize][j + halfSize];
                }
            }
            output[y][x] = static_cast<int>(sum);
        }
    }
    return output;
}

// Function to apply Bilateral filter
vector<vector<int>> applyBilateralFilter(const vector<vector<int>> &image, int width, int height, double sigma_s, double sigma_r, int kernelSize) {
    int halfSize = kernelSize / 2;
    vector<vector<int>> output(height, vector<int>(width));

// Apply Bilateral filter
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum = 0.0, normFactor = 0.0;
            for (int i = -halfSize; i <= halfSize; i++) {
                for (int j = -halfSize; j <= halfSize; j++) {
                    int yIndex = y + i;
                    int xIndex = x + j;

                    if (yIndex < 0) yIndex = -yIndex;
                    if (yIndex >= height) yIndex = 2 * height - yIndex - 2;
                    if (xIndex < 0) xIndex = -xIndex;
                    if (xIndex >= width) xIndex = 2 * width - xIndex - 2;

                    double spatialWeight = exp(-((i * i + j * j) / (2 * sigma_s * sigma_s)));
                    double rangeWeight = exp(-pow(image[yIndex][xIndex] - image[y][x], 2) / (2 * sigma_r * sigma_r));
                    double weight = spatialWeight * rangeWeight;
                    sum += weight * image[yIndex][xIndex];
                    normFactor += weight;
                }
            }
            output[y][x] = static_cast<int>(sum / normFactor);
        }
    }
    return output;
}

int main() {
    string inputFile = "C:\\Users\\usaal\\Downloads\\test-img.pgm";
    string outputGaussian = "C:\\Users\\usaal\\Desktop\\output_gaussian.pgm";
    string outputBilateral = "C:\\Users\\usaal\\Desktop\\output_bilateral.pgm";
    
    vector<vector<int>> image;
    int width, height, maxVal;
    
    if (!readPGM(inputFile, image, width, height, maxVal)) {
        return 1;
    }

    int kernelSize;
    cout << "Enter kernel size (odd number, >= 5): ";
    cin >> kernelSize;
    if (kernelSize % 2 == 0) kernelSize++;

// Apply Gaussian filter
    double sigma_x = 1.0, sigma_y = 1.5;
    vector<vector<int>> gaussianFiltered = applyGaussianFilter(image, width, height, sigma_x, sigma_y, kernelSize);

// Apply Bilateral filter
    double sigma_s = 2.0, sigma_r = 50.0;
    vector<vector<int>> bilateralFiltered = applyBilateralFilter(image, width, height, sigma_s, sigma_r, kernelSize);
    
    writePGM(outputGaussian, gaussianFiltered, width, height, maxVal);
    writePGM(outputBilateral, bilateralFiltered, width, height, maxVal);
    
    cout << "Filtering complete. Output images saved." << endl;
    return 0;
}