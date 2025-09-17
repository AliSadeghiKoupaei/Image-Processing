#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>

#define PI 3.14159265358979323846

using namespace std;
using Complex = complex<double>;

void readPGM(const string &filename, vector<vector<double>> &img, int &M, int &N) {
    ifstream infile(filename, ios::binary);
    string header;
    infile >> header >> N >> M;
    int maxval;
    infile >> maxval;
    infile.ignore();
    
    img.resize(M, vector<double>(N));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            img[i][j] = infile.get();
}

void writePGM(const string &filename, const vector<vector<double>> &img, int M, int N) {
    ofstream outfile(filename, ios::binary);
    outfile << "P5\n" << N << " " << M << "\n255\n";
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            outfile.put(static_cast<unsigned char>(min(255.0, max(0.0, img[i][j]))));
}

vector<Complex> dft1d(const vector<Complex> &signal) {
    int N = signal.size();
    vector<Complex> result(N);
    for (int k = 0; k < N; ++k)
        for (int n = 0; n < N; ++n)
            result[k] += signal[n] * exp(Complex(0, -2 * PI * k * n / N));
    return result;
}

// Perform 2D DFT on image
void dft2d(const vector<vector<double>> &img, vector<vector<Complex>> &F, int P, int Q) {
    vector<vector<Complex>> temp(P, vector<Complex>(Q));
    for (int i = 0; i < P; ++i) {
        vector<Complex> row(Q);
        for (int j = 0; j < Q; ++j)
            row[j] = img[i][j] * pow(-1, i + j);
        temp[i] = dft1d(row);
    }
    for (int j = 0; j < Q; ++j) {
        vector<Complex> col(P);
        for (int i = 0; i < P; ++i)
            col[i] = temp[i][j];
        col = dft1d(col);
        for (int i = 0; i < P; ++i)
            F[i][j] = col[i];
    }
}

// Ideal Low Pass Filter
void idealLowPass(vector<vector<Complex>> &F, int P, int Q, double D0) {
    for (int u = 0; u < P; ++u)
        for (int v = 0; v < Q; ++v) {
            double D = sqrt(pow(u - P / 2.0, 2) + pow(v - Q / 2.0, 2));
            if (D > D0)
                F[u][v] = 0;
        }
}

//  2D inverse DFT
void idft2d(const vector<vector<Complex>> &F, vector<vector<double>> &out, int P, int Q, int M, int N) {
    vector<vector<Complex>> temp(P, vector<Complex>(Q));
    for (int j = 0; j < Q; ++j) {
        vector<Complex> col(P);
        for (int i = 0; i < P; ++i)
            col[i] = F[i][j];
        for (int n = 0; n < P; ++n) {
            Complex sum(0, 0);
            for (int k = 0; k < P; ++k)
                sum += col[k] * exp(Complex(0, 2 * PI * k * n / P));
            temp[n][j] = sum / Complex(P);
        }
    }
    for (int i = 0; i < P; ++i) {
        vector<Complex> row(Q);
        for (int j = 0; j < Q; ++j)
            row[j] = temp[i][j];
        for (int n = 0; n < Q; ++n) {
            Complex sum(0, 0);
            for (int k = 0; k < Q; ++k)
                sum += row[k] * exp(Complex(0, 2 * PI * k * n / Q));
            double val = real(sum) / Q;
            val *= pow(-1, i + n);
            if (i < M && n < N)
                out[i][n] = val;
        }
    }
}

int main() {
    int M, N;
    vector<vector<double>> img;
    readPGM("Knee.pgm", img, M, N);

    int P = 2 * M - 1;
    int Q = 2 * N - 1;

    vector<vector<double>> padded(P, vector<double>(Q, 0.0));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            padded[i][j] = img[i][j];

    vector<vector<Complex>> F(P, vector<Complex>(Q));
    dft2d(padded, F, P, Q);

    for (double D0 : {10, 160}) {
        vector<vector<Complex>> F_filtered = F;
        idealLowPass(F_filtered, P, Q, D0);

        vector<vector<double>> output(M, vector<double>(N));
        idft2d(F_filtered, output, P, Q, M, N);

        string filename = "output_D0_" + to_string((int)D0) + ".pgm";
        writePGM(filename, output, M, N);
        cout << "Saved: " << filename << endl;
    }

    return 0;
}