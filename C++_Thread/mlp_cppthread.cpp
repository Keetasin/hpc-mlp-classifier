#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
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

const int NUM_THREADS = 8; // กำหนดจำนวน Threads 

// ==========================================
// 2. Simple Thread Pool 
// ==========================================
class SimpleThreadPool {
    vector<thread> threads;
    function<void(int)> current_task;
    atomic<int> completed_threads;
    atomic<int> generation;
    mutex mtx;
    condition_variable cv_start, cv_end;
    bool stop = false;
    int num_threads;

public:
    SimpleThreadPool(int n) : num_threads(n), completed_threads(0), generation(0) {
        for (int i = 0; i < n; ++i) {
            threads.emplace_back([this, i] {
                int my_gen = 0;
                while (true) {
                    unique_lock<mutex> lock(mtx);
                    cv_start.wait(lock, [this, my_gen] { return stop || generation > my_gen; });
                    if (stop) return;
                    my_gen = generation;
                    lock.unlock();

                    current_task(i);

                    if (++completed_threads == num_threads) {
                        unique_lock<mutex> lock_end(mtx);
                        cv_end.notify_one();
                    }
                }
            });
        }
    }

    ~SimpleThreadPool() {
        {
            unique_lock<mutex> lock(mtx);
            stop = true;
            generation++;
        }
        cv_start.notify_all();
        for (auto& t : threads) t.join();
    }

    void execute(function<void(int)> task) {
        {
            unique_lock<mutex> lock(mtx);
            current_task = task;
            completed_threads = 0;
            generation++;
        }
        cv_start.notify_all();

        unique_lock<mutex> lock(mtx);
        cv_end.wait(lock, [this] { return completed_threads == num_threads; });
    }
};

SimpleThreadPool* pool = nullptr;

// ==========================================
// 3. Data Loading Functions (IDX Format)
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
// 4. Main Training Loop
// ==========================================
int main() {
    pool = new SimpleThreadPool(NUM_THREADS); 

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

    int t_correct[128];
    float t_loss[128];

    // ==========================================
    // --- EPOCH LOOP ---
    // ==========================================
    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        auto train_start = chrono::high_resolution_clock::now();

        int correct_train = 0;
        float train_loss_total = 0.0f;

        for (int b = 0; b < train_batches; b++) {
            // Load batch to L1/L2 cache
            memcpy(d_X, h_train_X + b * BATCH_SIZE * INPUT_SIZE, BATCH_SIZE * INPUT_SIZE * sizeof(float));
            memcpy(d_Y, h_train_Y + b * BATCH_SIZE, BATCH_SIZE * sizeof(float));

            for(int i=0; i<128; i++) { t_correct[i] = 0; t_loss[i] = 0.0f; }

            // TASK 1: MEGA-FUSION (Forward 1 + Forward 2 + Loss + Backward 1)
            pool->execute([&](int t_id) {
                int chunk = BATCH_SIZE / NUM_THREADS;
                int start = t_id * chunk;
                int end = (t_id == NUM_THREADS - 1) ? BATCH_SIZE : start + chunk;

                int local_correct = 0;
                float local_loss = 0.0f;
                float inv_batch = 1.0f / BATCH_SIZE;

                for (int i = start; i < end; ++i) {
                    // Forward 1: X * W1 -> H
                    memset(&d_H[i * HIDDEN_SIZE], 0, HIDDEN_SIZE * sizeof(float));
                    for (int k = 0; k < INPUT_SIZE; ++k) {
                        float x_val = d_X[i * INPUT_SIZE + k];
                        #pragma GCC ivdep
                        for (int j = 0; j < HIDDEN_SIZE; ++j) {
                            d_H[i * HIDDEN_SIZE + j] += x_val * d_W1[k * HIDDEN_SIZE + j];
                        }
                    }
                    #pragma GCC ivdep
                    for (int j = 0; j < HIDDEN_SIZE; ++j) {
                        d_H[i * HIDDEN_SIZE + j] = fmaxf(0.0f, d_H[i * HIDDEN_SIZE + j]);
                    }

                    // Forward 2: H * W2 -> O
                    memset(&d_O[i * OUTPUT_SIZE], 0, OUTPUT_SIZE * sizeof(float));
                    for (int k = 0; k < HIDDEN_SIZE; ++k) {
                        float h_val = d_H[i * HIDDEN_SIZE + k];
                        #pragma GCC ivdep
                        for (int j = 0; j < OUTPUT_SIZE; ++j) {
                            d_O[i * OUTPUT_SIZE + j] += h_val * d_W2[k * OUTPUT_SIZE + j];
                        }
                    }

                    // Softmax & Loss
                    float* out_row = &d_O[i * OUTPUT_SIZE];
                    float max_val = out_row[0];
                    for (int j = 1; j < OUTPUT_SIZE; ++j) max_val = fmaxf(max_val, out_row[j]);
                    
                    float sum = 0.0f;
                    for (int j = 0; j < OUTPUT_SIZE; ++j) {
                        out_row[j] = expf(out_row[j] - max_val);
                        sum += out_row[j];
                    }
                    float inv_sum = 1.0f / sum;

                    int y_true = (int)d_Y[i];
                    int best_class = 0;
                    float max_prob = -1.0f;
                    float prob_correct = 1e-7f;
                    float* dz2_row = &d_dZ2[i * OUTPUT_SIZE];

                    #pragma GCC ivdep
                    for (int j = 0; j < OUTPUT_SIZE; ++j) {
                        out_row[j] *= inv_sum;
                        if (out_row[j] > max_prob) { max_prob = out_row[j]; best_class = j; }
                        if (j == y_true) prob_correct = fmaxf(out_row[j], 1e-7f);
                        dz2_row[j] = (out_row[j] - (j == y_true ? 1.0f : 0.0f)) * inv_batch;
                    }

                    if (best_class == y_true) local_correct++;
                    local_loss += -logf(prob_correct);

                    // Backward 1: dZ2 * W2^T -> dH -> Relu Deriv -> dZ1
                    float* dz1_row = &d_dZ1[i * HIDDEN_SIZE];
                    const float* h_row = &d_H[i * HIDDEN_SIZE];
                    for (int j = 0; j < HIDDEN_SIZE; ++j) {
                        float sum_dz1 = 0.0f;
                        const float* w2_row = &d_W2[j * OUTPUT_SIZE];
                        #pragma GCC ivdep
                        for (int k = 0; k < OUTPUT_SIZE; ++k) {
                            sum_dz1 += dz2_row[k] * w2_row[k];
                        }
                        dz1_row[j] = (h_row[j] > 0.0f) ? sum_dz1 : 0.0f; 
                    }
                }
                t_correct[t_id * 16] = local_correct;
                t_loss[t_id * 16] = local_loss;
            });

            for (int i = 0; i < NUM_THREADS; ++i) {
                correct_train += t_correct[i * 16];
                train_loss_total += t_loss[i * 16];
            }

            // TASK 2: Update W2 (Memory Elision - วาง Array ลง Stack ตรงๆ)
            pool->execute([&](int t_id) {
                int chunk = HIDDEN_SIZE / NUM_THREADS;
                int start = t_id * chunk;
                int end = (t_id == NUM_THREADS - 1) ? HIDDEN_SIZE : start + chunk;
                
                for (int row = start; row < end; ++row) {
                    float dw2_row[OUTPUT_SIZE] = {0}; 
                    for (int i = 0; i < BATCH_SIZE; ++i) {
                        float h_val = d_H[i * HIDDEN_SIZE + row];
                        const float* dz2_row = &d_dZ2[i * OUTPUT_SIZE];
                        #pragma GCC ivdep
                        for (int col = 0; col < OUTPUT_SIZE; ++col) {
                            dw2_row[col] += h_val * dz2_row[col];
                        }
                    }
                    #pragma GCC ivdep
                    for (int col = 0; col < OUTPUT_SIZE; ++col) {
                        d_W2[row * OUTPUT_SIZE + col] -= LEARNING_RATE * dw2_row[col];
                    }
                }
            });

            // TASK 3: Update W1 (Memory Elision - วาง Array ลง Stack ตรงๆ)
            pool->execute([&](int t_id) {
                int chunk = INPUT_SIZE / NUM_THREADS;
                int start = t_id * chunk;
                int end = (t_id == NUM_THREADS - 1) ? INPUT_SIZE : start + chunk;
                
                for (int row = start; row < end; ++row) {
                    float dw1_row[HIDDEN_SIZE] = {0}; 
                    for (int i = 0; i < BATCH_SIZE; ++i) {
                        float x_val = d_X[i * INPUT_SIZE + row];
                        const float* dz1_row = &d_dZ1[i * HIDDEN_SIZE];
                        #pragma GCC ivdep
                        for (int col = 0; col < HIDDEN_SIZE; ++col) {
                            dw1_row[col] += x_val * dz1_row[col];
                        }
                    }
                    #pragma GCC ivdep
                    for (int col = 0; col < HIDDEN_SIZE; ++col) {
                        d_W1[row * HIDDEN_SIZE + col] -= LEARNING_RATE * dw1_row[col];
                    }
                }
            });
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

            for(int i=0; i<128; i++) { t_correct[i] = 0; t_loss[i] = 0.0f; }

            pool->execute([&](int t_id) {
                int chunk = BATCH_SIZE / NUM_THREADS;
                int start = t_id * chunk;
                int end = (t_id == NUM_THREADS - 1) ? BATCH_SIZE : start + chunk;

                int local_correct = 0;
                float local_loss = 0.0f;

                for (int i = start; i < end; ++i) {
                    memset(&d_H[i * HIDDEN_SIZE], 0, HIDDEN_SIZE * sizeof(float));
                    for (int k = 0; k < INPUT_SIZE; ++k) {
                        float a_val = d_X[i * INPUT_SIZE + k];
                        #pragma GCC ivdep
                        for (int j = 0; j < HIDDEN_SIZE; ++j) {
                            d_H[i * HIDDEN_SIZE + j] += a_val * d_W1[k * HIDDEN_SIZE + j];
                        }
                    }
                    #pragma GCC ivdep
                    for (int j = 0; j < HIDDEN_SIZE; ++j) {
                        d_H[i * HIDDEN_SIZE + j] = fmaxf(0.0f, d_H[i * HIDDEN_SIZE + j]);
                    }

                    memset(&d_O[i * OUTPUT_SIZE], 0, OUTPUT_SIZE * sizeof(float));
                    for (int k = 0; k < HIDDEN_SIZE; ++k) {
                        float h_val = d_H[i * HIDDEN_SIZE + k];
                        #pragma GCC ivdep
                        for (int j = 0; j < OUTPUT_SIZE; ++j) {
                            d_O[i * OUTPUT_SIZE + j] += h_val * d_W2[k * OUTPUT_SIZE + j];
                        }
                    }

                    float* out_row = &d_O[i * OUTPUT_SIZE];
                    float max_val = out_row[0];
                    for (int j = 1; j < OUTPUT_SIZE; ++j) max_val = fmaxf(max_val, out_row[j]);
                    
                    float sum = 0.0f;
                    for (int j = 0; j < OUTPUT_SIZE; ++j) {
                        out_row[j] = expf(out_row[j] - max_val);
                        sum += out_row[j];
                    }
                    float inv_sum = 1.0f / sum;

                    int y_true = (int)d_Y[i];
                    int best_class = 0;
                    float max_prob = -1.0f;
                    float prob_correct = 1e-7f;

                    #pragma GCC ivdep
                    for (int j = 0; j < OUTPUT_SIZE; ++j) {
                        out_row[j] *= inv_sum;
                        if (out_row[j] > max_prob) { max_prob = out_row[j]; best_class = j; }
                        if (j == y_true) prob_correct = fmaxf(out_row[j], 1e-7f);
                    }
                    if (best_class == y_true) local_correct++;
                    local_loss += -logf(prob_correct);
                }
                t_correct[t_id * 16] = local_correct;
                t_loss[t_id * 16] = local_loss;
            });

            for (int i = 0; i < NUM_THREADS; ++i) {
                correct_val += t_correct[i * 16];
                val_loss_total += t_loss[i * 16];
            }
        }
        
        final_val_acc = (float)correct_val / (val_batches * BATCH_SIZE) * 100.0f;
        final_val_loss = val_loss_total / (val_batches * BATCH_SIZE);

        printf("Epoch [%d/%d] - Train Acc: %.3f %% - Val Acc: %.3f %% - Train Loss: %.4f - Val Loss: %.4f\n",
               epoch + 1, EPOCHS, final_train_acc, final_val_acc, final_train_loss, final_val_loss);
    }

    // ==========================================
    // --- 3. FINAL TEST PHASE ---
    // ==========================================
    int correct_test = 0;
    float test_loss_total = 0.0f;

    for (int b = 0; b < test_batches; b++) {
        memcpy(d_X, h_test_X + b * BATCH_SIZE * INPUT_SIZE, BATCH_SIZE * INPUT_SIZE * sizeof(float));
        memcpy(d_Y, h_test_Y + b * BATCH_SIZE, BATCH_SIZE * sizeof(float));

        for(int i=0; i<128; i++) { t_correct[i] = 0; t_loss[i] = 0.0f; }

        pool->execute([&](int t_id) {
            int chunk = BATCH_SIZE / NUM_THREADS;
            int start = t_id * chunk;
            int end = (t_id == NUM_THREADS - 1) ? BATCH_SIZE : start + chunk;

            int local_correct = 0;
            float local_loss = 0.0f;

            for (int i = start; i < end; ++i) {
                memset(&d_H[i * HIDDEN_SIZE], 0, HIDDEN_SIZE * sizeof(float));
                for (int k = 0; k < INPUT_SIZE; ++k) {
                    float a_val = d_X[i * INPUT_SIZE + k];
                    #pragma GCC ivdep
                    for (int j = 0; j < HIDDEN_SIZE; ++j) {
                        d_H[i * HIDDEN_SIZE + j] += a_val * d_W1[k * HIDDEN_SIZE + j];
                    }
                }
                #pragma GCC ivdep
                for (int j = 0; j < HIDDEN_SIZE; ++j) {
                    d_H[i * HIDDEN_SIZE + j] = fmaxf(0.0f, d_H[i * HIDDEN_SIZE + j]);
                }

                memset(&d_O[i * OUTPUT_SIZE], 0, OUTPUT_SIZE * sizeof(float));
                for (int k = 0; k < HIDDEN_SIZE; ++k) {
                    float h_val = d_H[i * HIDDEN_SIZE + k];
                    #pragma GCC ivdep
                    for (int j = 0; j < OUTPUT_SIZE; ++j) {
                        d_O[i * OUTPUT_SIZE + j] += h_val * d_W2[k * OUTPUT_SIZE + j];
                    }
                }

                float* out_row = &d_O[i * OUTPUT_SIZE];
                float max_val = out_row[0];
                for (int j = 1; j < OUTPUT_SIZE; ++j) max_val = fmaxf(max_val, out_row[j]);
                
                float sum = 0.0f;
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    out_row[j] = expf(out_row[j] - max_val);
                    sum += out_row[j];
                }
                float inv_sum = 1.0f / sum;

                int y_true = (int)d_Y[i];
                int best_class = 0;
                float max_prob = -1.0f;
                float prob_correct = 1e-7f;

                #pragma GCC ivdep
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    out_row[j] *= inv_sum;
                    if (out_row[j] > max_prob) { max_prob = out_row[j]; best_class = j; }
                    if (j == y_true) prob_correct = fmaxf(out_row[j], 1e-7f);
                }
                if (best_class == y_true) local_correct++;
                local_loss += -logf(prob_correct);
            }
            t_correct[t_id * 16] = local_correct;
            t_loss[t_id * 16] = local_loss;
        });

        for (int i = 0; i < NUM_THREADS; ++i) {
            correct_test += t_correct[i * 16];
            test_loss_total += t_loss[i * 16];
        }
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

    delete pool; 
    delete[] d_X; delete[] d_Y; delete[] d_H; delete[] d_O;
    delete[] d_dZ1; delete[] d_dZ2; 
    delete[] h_train_X; delete[] h_train_Y; delete[] h_test_X; delete[] h_test_Y;
    delete[] h_W1; delete[] h_W2;

    return 0;
}