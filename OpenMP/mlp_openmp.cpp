#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cstring> 

using namespace std;

// ==========================================
// 1. Hyperparameters & Network Architecture
// ==========================================
#define INPUT_SIZE 784   // 28x28 pixels
#define HIDDEN_SIZE 256  // Nodes in Hidden Layer
#define OUTPUT_SIZE 10   // Classes (Fashion-MNIST)
#define BATCH_SIZE 256   // Mini-batch size
#define EPOCHS 100       // จำนวนรอบในการเทรน
#define LEARNING_RATE 0.05f // อัตราการเรียนรู้

// ==========================================
// 2. Data Loading Functions (IDX Format)
// ==========================================
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_images(string filename, float*& images, int& num_images) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_images, sizeof(num_images));
        num_images = reverseInt(num_images);
        file.read((char*)&n_rows, sizeof(n_rows)); file.read((char*)&n_cols, sizeof(n_cols));

        int image_size = INPUT_SIZE;
        images = new float[num_images * image_size];
        unsigned char* temp = new unsigned char[num_images * image_size];
        file.read((char*)temp, num_images * image_size);
        for (int i = 0; i < num_images * image_size; i++) images[i] = (float)temp[i] / 255.0f;
        delete[] temp;
    } else { cout << "Cannot open file: " << filename << endl; exit(1); }
}

void read_mnist_labels(string filename, float*& labels, int& num_labels) {
    ifstream file(filename, ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        file.read((char*)&num_labels, sizeof(num_labels));
        num_labels = reverseInt(num_labels);

        labels = new float[num_labels];
        unsigned char* temp = new unsigned char[num_labels];
        file.read((char*)temp, num_labels);
        for (int i = 0; i < num_labels; i++) labels[i] = (float)temp[i];
        delete[] temp;
    } else { cout << "Cannot open file: " << filename << endl; exit(1); }
}

// ==========================================
// 3. FUSED OpenMP Kernels 
// ==========================================

// Fused Forward 1: X * W1 -> H -> Relu
void forward1(const float* __restrict__ X, const float* __restrict__ W1, float* __restrict__ H) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < BATCH_SIZE; ++i) {
        memset(&H[i * HIDDEN_SIZE], 0, HIDDEN_SIZE * sizeof(float));
        for (int k = 0; k < INPUT_SIZE; ++k) {
            float x_val = X[i * INPUT_SIZE + k];
            #pragma omp simd
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                H[i * HIDDEN_SIZE + j] += x_val * W1[k * HIDDEN_SIZE + j];
            }
        }
        #pragma omp simd
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            H[i * HIDDEN_SIZE + j] = H[i * HIDDEN_SIZE + j] > 0.0f ? H[i * HIDDEN_SIZE + j] : 0.0f;
        }
    }
}

// Fused Forward 2 (Train): H * W2 -> O -> Softmax -> Eval -> dZ2
void forward2_train(const float* __restrict__ H, const float* __restrict__ W2, float* __restrict__ O, 
                    const float* __restrict__ Y, float* __restrict__ dZ2, float& total_loss, int& total_correct) {
    int local_correct = 0;
    float local_loss = 0.0f;
    float inv_batch = 1.0f / BATCH_SIZE;

    #pragma omp parallel for reduction(+:local_correct, local_loss) schedule(static)
    for (int i = 0; i < BATCH_SIZE; ++i) {
        float* o_row = &O[i * OUTPUT_SIZE];
        memset(o_row, 0, OUTPUT_SIZE * sizeof(float));
        
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            float h_val = H[i * HIDDEN_SIZE + k];
            #pragma omp simd
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                o_row[j] += h_val * W2[k * OUTPUT_SIZE + j];
            }
        }
        
        float max_val = o_row[0];
        for (int j = 1; j < OUTPUT_SIZE; ++j) max_val = max_val > o_row[j] ? max_val : o_row[j];
        
        float sum = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            o_row[j] = expf(o_row[j] - max_val);
            sum += o_row[j];
        }
        float inv_sum = 1.0f / sum;
        #pragma omp simd
        for (int j = 0; j < OUTPUT_SIZE; ++j) o_row[j] *= inv_sum;

        int y_true = (int)Y[i];
        int best_class = 0;
        float max_prob = o_row[0];
        float prob_correct = 1e-7f;
        float* dz2_row = &dZ2[i * OUTPUT_SIZE];
        
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            if (o_row[j] > max_prob) { max_prob = o_row[j]; best_class = j; }
            if (j == y_true) prob_correct = o_row[j] > 1e-7f ? o_row[j] : 1e-7f;
            dz2_row[j] = (o_row[j] - (j == y_true ? 1.0f : 0.0f)) * inv_batch;
        }
        if (best_class == y_true) local_correct++;
        local_loss += -logf(prob_correct);
    }
    total_correct += local_correct;
    total_loss += local_loss;
}

// Fused Forward 2 (Validation/Test)
void forward2_eval(const float* __restrict__ H, const float* __restrict__ W2, float* __restrict__ O, 
                   const float* __restrict__ Y, float& total_loss, int& total_correct) {
    int local_correct = 0;
    float local_loss = 0.0f;

    #pragma omp parallel for reduction(+:local_correct, local_loss) schedule(static)
    for (int i = 0; i < BATCH_SIZE; ++i) {
        float* o_row = &O[i * OUTPUT_SIZE];
        memset(o_row, 0, OUTPUT_SIZE * sizeof(float));
        
        for (int k = 0; k < HIDDEN_SIZE; ++k) {
            float h_val = H[i * HIDDEN_SIZE + k];
            #pragma omp simd
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                o_row[j] += h_val * W2[k * OUTPUT_SIZE + j];
            }
        }
        
        float max_val = o_row[0];
        for (int j = 1; j < OUTPUT_SIZE; ++j) max_val = max_val > o_row[j] ? max_val : o_row[j];
        
        float sum = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            o_row[j] = expf(o_row[j] - max_val);
            sum += o_row[j];
        }
        float inv_sum = 1.0f / sum;
        #pragma omp simd
        for (int j = 0; j < OUTPUT_SIZE; ++j) o_row[j] *= inv_sum;

        int y_true = (int)Y[i];
        int best_class = 0;
        float max_prob = o_row[0];
        float prob_correct = 1e-7f;
        
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            if (o_row[j] > max_prob) { max_prob = o_row[j]; best_class = j; }
            if (j == y_true) prob_correct = o_row[j] > 1e-7f ? o_row[j] : 1e-7f;
        }
        if (best_class == y_true) local_correct++;
        local_loss += -logf(prob_correct);
    }
    total_correct += local_correct;
    total_loss += local_loss;
}

// Fused Backward 1: dZ2 * W2^T -> dH -> ReLU Deriv -> dZ1
void backward1(const float* __restrict__ dZ2, const float* __restrict__ W2, const float* __restrict__ H, float* __restrict__ dZ1) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < BATCH_SIZE; ++i) {
        const float* dz2_row = &dZ2[i * OUTPUT_SIZE];
        float* dz1_row = &dZ1[i * HIDDEN_SIZE];
        const float* h_row = &H[i * HIDDEN_SIZE];
        
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            float sum = 0.0f;
            const float* w2_row = &W2[j * OUTPUT_SIZE];
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < OUTPUT_SIZE; ++k) {
                sum += dz2_row[k] * w2_row[k]; 
            }
            dz1_row[j] = (h_row[j] > 0.0f) ? sum : 0.0f; 
        }
    }
}

// Fused Backward 2: H^T * dZ2 -> dW2 -> Update W2 
void update_W2(const float* __restrict__ H, const float* __restrict__ dZ2, float* __restrict__ W2) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < HIDDEN_SIZE; ++row) {
        float dw2_row[OUTPUT_SIZE] = {0}; 
        for (int i = 0; i < BATCH_SIZE; ++i) {
            float h_val = H[i * HIDDEN_SIZE + row];
            const float* dz2_row = &dZ2[i * OUTPUT_SIZE];
            #pragma omp simd
            for (int col = 0; col < OUTPUT_SIZE; ++col) {
                dw2_row[col] += h_val * dz2_row[col];
            }
        }
        #pragma omp simd
        for (int col = 0; col < OUTPUT_SIZE; ++col) {
            W2[row * OUTPUT_SIZE + col] -= LEARNING_RATE * dw2_row[col];
        }
    }
}

// Fused Backward 3: X^T * dZ1 -> dW1 -> Update W1
void update_W1(const float* __restrict__ X, const float* __restrict__ dZ1, float* __restrict__ W1) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < INPUT_SIZE; ++row) {
        float dw1_row[HIDDEN_SIZE] = {0}; 
        for (int i = 0; i < BATCH_SIZE; ++i) {
            float x_val = X[i * INPUT_SIZE + row];
            const float* dz1_row = &dZ1[i * HIDDEN_SIZE];
            #pragma omp simd
            for (int col = 0; col < HIDDEN_SIZE; ++col) {
                dw1_row[col] += x_val * dz1_row[col];
            }
        }
        #pragma omp simd
        for (int col = 0; col < HIDDEN_SIZE; ++col) {
            W1[row * HIDDEN_SIZE + col] -= LEARNING_RATE * dw1_row[col];
        }
    }
}

// ==========================================
// 4. Main Training Loop
// ==========================================
int main() {
    float *h_train_X, *h_train_Y, *h_test_X, *h_test_Y;
    int total_train_size, train_label_size, test_size, test_label_size;

    cout << "Loading datasets..." << endl;
    read_mnist_images("train-images-idx3-ubyte", h_train_X, total_train_size);
    read_mnist_labels("train-labels-idx1-ubyte", h_train_Y, train_label_size);
    read_mnist_images("t10k-images-idx3-ubyte", h_test_X, test_size);
    read_mnist_labels("t10k-labels-idx1-ubyte", h_test_Y, test_label_size);
    if(total_train_size == 0 || test_size == 0 || total_train_size != train_label_size) return 1;

    int train_samples = 50000;
    int val_samples = total_train_size - train_samples;
    int train_batches = train_samples / BATCH_SIZE;
    int val_batches = val_samples / BATCH_SIZE;
    int test_batches = test_size / BATCH_SIZE;

    cout << "Train Size: " << train_samples << " | Val Size: " << val_samples << " | Test Size: " << test_size << endl;

    float *h_W1 = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *h_W2 = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) h_W1[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / INPUT_SIZE);
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) h_W2[i] = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / HIDDEN_SIZE);

    float *d_X = new float[BATCH_SIZE * INPUT_SIZE];
    float *d_Y = new float[BATCH_SIZE];
    
    float *d_W1 = h_W1;
    float *d_W2 = h_W2;
    float *d_H = new float[BATCH_SIZE * HIDDEN_SIZE];
    float *d_O = new float[BATCH_SIZE * OUTPUT_SIZE];
    float *d_dZ1 = new float[BATCH_SIZE * HIDDEN_SIZE];
    float *d_dZ2 = new float[BATCH_SIZE * OUTPUT_SIZE];

    cout << "\nStarting Training ..." << endl;

    double total_train_time = 0.0;
    float final_train_acc = 0.0f, final_val_acc = 0.0f, final_test_acc = 0.0f;
    float final_train_loss = 0.0f, final_val_loss = 0.0f, final_test_loss = 0.0f;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        auto train_start = chrono::high_resolution_clock::now();

        int correct_train = 0;
        float train_loss_total = 0.0f;

        for (int b = 0; b < train_batches; b++) {
            memcpy(d_X, h_train_X + b * BATCH_SIZE * INPUT_SIZE, BATCH_SIZE * INPUT_SIZE * sizeof(float));
            memcpy(d_Y, h_train_Y + b * BATCH_SIZE, BATCH_SIZE * sizeof(float));

            forward1(d_X, d_W1, d_H);
            forward2_train(d_H, d_W2, d_O, d_Y, d_dZ2, train_loss_total, correct_train);
            
            backward1(d_dZ2, d_W2, d_H, d_dZ1);
            
            update_W2(d_H, d_dZ2, d_W2);
            update_W1(d_X, d_dZ1, d_W1);
        }

        auto train_end = chrono::high_resolution_clock::now();
        chrono::duration<double> epoch_time = train_end - train_start;
        total_train_time += epoch_time.count();

        final_train_acc = (float)correct_train / (train_batches * BATCH_SIZE) * 100.0f;
        final_train_loss = train_loss_total / (train_batches * BATCH_SIZE);

        // --- VALIDATION PHASE ---
        int correct_val = 0;
        float val_loss_total = 0.0f;

        for (int b = 0; b < val_batches; b++) {
            memcpy(d_X, h_train_X + (train_samples + b * BATCH_SIZE) * INPUT_SIZE, BATCH_SIZE * INPUT_SIZE * sizeof(float));
            memcpy(d_Y, h_train_Y + (train_samples + b * BATCH_SIZE), BATCH_SIZE * sizeof(float));

            forward1(d_X, d_W1, d_H);
            forward2_eval(d_H, d_W2, d_O, d_Y, val_loss_total, correct_val);
        }
        
        final_val_acc = (float)correct_val / (val_batches * BATCH_SIZE) * 100.0f;
        final_val_loss = val_loss_total / (val_batches * BATCH_SIZE);

        printf("Epoch [%d/%d] - Train Acc: %.3f %% - Val Acc: %.3f %% - Train Loss: %.4f - Val Loss: %.4f\n",
               epoch + 1, EPOCHS, final_train_acc, final_val_acc, final_train_loss, final_val_loss);
    }

    // --- FINAL TEST PHASE ---
    int correct_test = 0;
    float test_loss_total = 0.0f;

    for (int b = 0; b < test_batches; b++) {
        memcpy(d_X, h_test_X + b * BATCH_SIZE * INPUT_SIZE, BATCH_SIZE * INPUT_SIZE * sizeof(float));
        memcpy(d_Y, h_test_Y + b * BATCH_SIZE, BATCH_SIZE * sizeof(float));

        forward1(d_X, d_W1, d_H);
        forward2_eval(d_H, d_W2, d_O, d_Y, test_loss_total, correct_test);
    }
    
    final_test_acc = (float)correct_test / (test_batches * BATCH_SIZE) * 100.0f;
    final_test_loss = test_loss_total / (test_batches * BATCH_SIZE);

    // --- GFLOPS Calculation ---
    double ops_per_batch = 2.0 * BATCH_SIZE * HIDDEN_SIZE * INPUT_SIZE +
                           2.0 * BATCH_SIZE * OUTPUT_SIZE * HIDDEN_SIZE +
                           2.0 * HIDDEN_SIZE * OUTPUT_SIZE * BATCH_SIZE +
                           2.0 * BATCH_SIZE * HIDDEN_SIZE * OUTPUT_SIZE +
                           2.0 * INPUT_SIZE * HIDDEN_SIZE * BATCH_SIZE;

    double total_ops = ops_per_batch * train_batches * EPOCHS;
    double gflops = (total_ops / total_train_time) / 1e9;

    // --- FINAL OUTPUT ---
    cout << "\n===============================================" << endl;
    cout << "                 FINAL RESULTS                 " << endl;
    cout << "===============================================" << endl;
    printf("%-15s | %-14s | %-10s\n", "Dataset Phase", "Accuracy (%)", "Loss");
    cout << "-----------------------------------------------" << endl;
    printf("%-15s | %-14.3f | %-10.4f\n", "Training", final_train_acc, final_train_loss);
    printf("%-15s | %-14.3f | %-10.4f\n", "Validation", final_val_acc, final_val_loss);
    printf("%-15s | %-14.3f | %-10.4f\n", "Testing", final_test_acc, final_test_loss);
    cout << "-----------------------------------------------" << endl;
    printf("%-20s : %.4f seconds\n", "Total Training Time", total_train_time);
    printf("%-20s : %.3f GFLOPS\n", "Throughput", gflops);
    cout << "===============================================" << endl;

    delete[] d_X; delete[] d_Y; delete[] d_H; delete[] d_O;
    delete[] d_dZ1; delete[] d_dZ2; 
    delete[] h_train_X; delete[] h_train_Y; delete[] h_test_X; delete[] h_test_Y;
    delete[] h_W1; delete[] h_W2;

    return 0;
}