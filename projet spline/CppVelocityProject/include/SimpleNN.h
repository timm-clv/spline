#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// Classe simple pour gérer une couche dense (Linear + ReLU optionnel)
class DenseLayer {
public:
    int inputs;
    int outputs;
    std::vector<float> weights; // Aplatit row-major
    std::vector<float> bias;
    bool use_relu;

    DenseLayer(int in, int out, bool relu) : inputs(in), outputs(out), use_relu(relu) {}

    // Propagation avant : y = ReLU(W*x + b)
    std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> y(outputs, 0.0f);
        
        for (int i = 0; i < outputs; ++i) {
            float sum = bias[i];
            for (int j = 0; j < inputs; ++j) {
                // Poids stockés en format [Out, In] comme PyTorch
                sum += weights[i * inputs + j] * x[j];
            }
            // Activation ReLU
            if (use_relu) {
                y[i] = (sum > 0.0f) ? sum : 0.0f;
            } else {
                y[i] = sum;
            }
        }
        return y;
    }
};

class SimpleNN {
public:
    std::vector<DenseLayer> layers;
    
    // Stats de normalisation
    std::vector<float> X_mean, X_std;
    float Y_mean, Y_std;

    bool load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[ERREUR] Impossible d'ouvrir " << filepath << std::endl;
            return false;
        }

        std::string line, key;
        
        // 1. Lecture Stats
        file >> key; // "STATS"
        X_mean.resize(6); X_std.resize(6);
        for(int i=0; i<6; ++i) file >> X_mean[i];
        for(int i=0; i<6; ++i) file >> X_std[i];
        file >> Y_mean >> Y_std;

        // 2. Lecture Couches
        // On suppose une structure fixe 6->64->64->32->1 pour simplifier la lecture ReLU
        // Mais on lit les tailles dynamiquement.
        while (file >> key) { 
            if (key == "LAYER") {
                int rows, cols;
                file >> rows >> cols;
                
                // La dernière couche (output 1) n'a pas de ReLU, les autres si.
                bool is_output = (rows == 1); 
                DenseLayer layer(cols, rows, !is_output);
                
                layer.weights.resize(rows * cols);
                for(int i=0; i<rows*cols; ++i) file >> layer.weights[i];
                
                layer.bias.resize(rows);
                for(int i=0; i<rows; ++i) file >> layer.bias[i];
                
                layers.push_back(layer);
            }
        }
        return true;
    }

    float predict(const std::vector<float>& input_raw) {
        // 1. Normalisation
        std::vector<float> x = input_raw;
        for(size_t i=0; i<x.size(); ++i) {
            x[i] = (x[i] - X_mean[i]) / (X_std[i] + 1e-6f);
        }

        // 2. Forward Pass
        for(const auto& layer : layers) {
            x = layer.forward(x);
        }

        // 3. Dénormalisation Sortie
        return x[0] * Y_std + Y_mean;
    }
};