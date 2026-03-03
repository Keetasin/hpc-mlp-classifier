import os
import time
import struct
import numpy as np
import pyspark
from pyspark import SparkConf, SparkContext
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. Hyperparameters & Network Architecture
# ==========================================
INPUT_SIZE = 784   # 28x28 pixels
HIDDEN_SIZE = 256  # Nodes in Hidden Layer
OUTPUT_SIZE = 10   # Classes (Fashion-MNIST)
BATCH_SIZE = 256   # Mini-batch size
EPOCHS = 100       # Number of epochs
LEARNING_RATE = 0.05 # Learning rate

# ==========================================
# 2. Data Loading Functions (IDX Format)
# ==========================================
def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images.astype(np.float32) / 255.0

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels.astype(np.float32)

# ==========================================
# 3. Neural Network Functions
# ==========================================
def relu(x):
    return np.maximum(0, x)

def relu_backward(dH, H):
    return dH * (H > 0)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def forward_pass(X, W1, W2):
    H = relu(np.dot(X, W1))
    O = softmax(np.dot(H, W2))
    return H, O

def compute_loss_and_correct(O, Y):
    batch_size = O.shape[0]
    predictions = np.argmax(O, axis=1)
    correct = np.sum(predictions == Y)
    
    # Clip probabilities to prevent log(0)
    probs = np.clip(O[np.arange(batch_size), Y.astype(int)], 1e-7, 1.0)
    loss = -np.sum(np.log(probs))
    return loss, correct

def backward_pass(X, Y, H, O, W2):
    batch_size = X.shape[0]
    
    Y_one_hot = np.zeros((batch_size, OUTPUT_SIZE))
    Y_one_hot[np.arange(batch_size), Y.astype(int)] = 1.0
    
    dZ2 = (O - Y_one_hot) / batch_size
    dW2 = np.dot(H.T, dZ2)
    dH = np.dot(dZ2, W2.T)
    
    dZ1 = relu_backward(dH, H)
    dW1 = np.dot(X.T, dZ1)
    
    return dW1, dW2

# ==========================================
# 4. Partition Functions (Local SGD)
# ==========================================
def train_partition(iterator, W1_bc, W2_bc, learning_rate):
    # Copy weights locally to update them inside the partition
    W1_local = np.copy(W1_bc.value)
    W2_local = np.copy(W2_bc.value)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_data in iterator:
        X_batch, Y_batch = batch_data
        
        # Forward & Backward Pass
        H, O = forward_pass(X_batch, W1_local, W2_local)
        loss, correct = compute_loss_and_correct(O, Y_batch)
        dW1, dW2 = backward_pass(X_batch, Y_batch, H, O, W2_local)
        
        # MINI-BATCH UPDATE: Update local weights immediately!
        W1_local -= learning_rate * dW1
        W2_local -= learning_rate * dW2
        
        total_loss += loss
        total_correct += correct
        total_samples += X_batch.shape[0]
        
    # Yield the updated weights and metrics back to the driver
    yield (W1_local, W2_local, total_loss, total_correct, total_samples)

def eval_partition(iterator, W1_bc, W2_bc):
    W1 = W1_bc.value
    W2 = W2_bc.value
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_data in iterator:
        X_batch, Y_batch = batch_data
        _, O = forward_pass(X_batch, W1, W2)
        loss, correct = compute_loss_and_correct(O, Y_batch)
        
        total_loss += loss
        total_correct += correct
        total_samples += X_batch.shape[0]
        
    yield (total_loss, total_correct, total_samples)

# ==========================================
# 5. Main Training Loop
# ==========================================
def main():
    conf = SparkConf().setAppName("MLP_PySpark_6610110425")
    sc = SparkContext(conf=conf)
    
    print("Loading datasets...")
    try:
        train_X = read_mnist_images("train-images-idx3-ubyte")
        train_Y = read_mnist_labels("train-labels-idx1-ubyte")
        test_X = read_mnist_images("t10k-images-idx3-ubyte")
        test_Y = read_mnist_labels("t10k-labels-idx1-ubyte")
    except Exception as e:
         print(f"Error loading data: {e}. Ensure the IDX files are in the current directory.")
         sc.stop()
         return

    total_train_size = train_X.shape[0]
    test_size = test_X.shape[0]
    
    train_samples = 50000
    val_samples = total_train_size - train_samples
    
    print(f"Train Size: {train_samples} | Val Size: {val_samples} | Test Size: {test_size}")

    X_train = train_X[:train_samples]
    Y_train = train_Y[:train_samples]
    X_val = train_X[train_samples:]
    Y_val = train_Y[train_samples:]
    
    def create_batches(X, Y, batch_size):
        batches = []
        for i in range(0, X.shape[0], batch_size):
            end = min(i + batch_size, X.shape[0])
            batches.append((X[i:end], Y[i:end]))
        return batches
        
    train_batches = create_batches(X_train, Y_train, BATCH_SIZE)
    val_batches = create_batches(X_val, Y_val, BATCH_SIZE)
    test_batches = create_batches(test_X, test_Y, BATCH_SIZE)

    np.random.seed(42)
    W1 = (np.random.rand(INPUT_SIZE, HIDDEN_SIZE) - 0.5) * np.sqrt(2.0 / INPUT_SIZE)
    W2 = (np.random.rand(HIDDEN_SIZE, OUTPUT_SIZE) - 0.5) * np.sqrt(2.0 / HIDDEN_SIZE)

    print("\nStarting Training...")
    
    total_train_time = 0.0
    
    num_train_batches = len(train_batches)
    num_partitions = min(8, num_train_batches) 
    
    # Cache RDDs
    train_rdd = sc.parallelize(train_batches, numSlices=num_partitions).cache()
    val_rdd = sc.parallelize(val_batches, numSlices=num_partitions).cache()
    test_rdd = sc.parallelize(test_batches, numSlices=num_partitions).cache()
    
    actual_train_partitions = train_rdd.getNumPartitions()

    # ==========================================
    # --- EPOCH LOOP ---
    # ==========================================
    for epoch in range(EPOCHS):
        W1_bc = sc.broadcast(W1)
        W2_bc = sc.broadcast(W2)
        
        train_start = time.time()

        # --- 1. TRAIN PHASE (Local SGD using mapPartitions) ---
        train_results = train_rdd.mapPartitions(lambda it: train_partition(it, W1_bc, W2_bc, LEARNING_RATE)) \
                                 .reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4]))
        
        sum_W1, sum_W2, total_loss, total_correct, total_samples = train_results
        
        # FEDERATED AVERAGING: Average the locally trained weights from all partitions
        W1 = sum_W1 / actual_train_partitions
        W2 = sum_W2 / actual_train_partitions

        train_end = time.time()
        total_train_time += (train_end - train_start)

        final_train_acc = (total_correct / total_samples) * 100.0
        final_train_loss = total_loss / total_samples

        W1_bc.unpersist()
        W2_bc.unpersist()

        # --- 2. VALIDATION PHASE ---
        W1_val_bc = sc.broadcast(W1)
        W2_val_bc = sc.broadcast(W2)
        
        val_results = val_rdd.mapPartitions(lambda it: eval_partition(it, W1_val_bc, W2_val_bc)) \
                             .reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
        
        val_total_loss, val_total_correct, val_total_samples = val_results
        
        final_val_acc = (val_total_correct / val_total_samples) * 100.0
        final_val_loss = val_total_loss / val_total_samples
        
        W1_val_bc.unpersist()
        W2_val_bc.unpersist()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Train Acc: {final_train_acc:.3f} % - Val Acc: {final_val_acc:.3f} % - Train Loss: {final_train_loss:.4f} - Val Loss: {final_val_loss:.4f}")

    # ==========================================
    # --- 3. FINAL TEST PHASE ---
    # ==========================================
    W1_test_bc = sc.broadcast(W1)
    W2_test_bc = sc.broadcast(W2)
    
    test_results = test_rdd.mapPartitions(lambda it: eval_partition(it, W1_test_bc, W2_test_bc)) \
                           .reduce(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
    
    test_total_loss, test_total_correct, test_total_samples = test_results
    
    final_test_acc = (test_total_correct / test_total_samples) * 100.0
    final_test_loss = test_total_loss / test_total_samples

    # --- GFLOPS Calculation ---
    ops_per_batch = 2.0 * BATCH_SIZE * HIDDEN_SIZE * INPUT_SIZE + \
                    2.0 * BATCH_SIZE * OUTPUT_SIZE * HIDDEN_SIZE + \
                    2.0 * HIDDEN_SIZE * OUTPUT_SIZE * BATCH_SIZE + \
                    2.0 * BATCH_SIZE * HIDDEN_SIZE * OUTPUT_SIZE + \
                    2.0 * INPUT_SIZE * HIDDEN_SIZE * BATCH_SIZE
                           
    total_ops = ops_per_batch * num_train_batches * EPOCHS
    gflops = (total_ops / total_train_time) / 1e9 if total_train_time > 0 else 0.0

    # --- FINAL OUTPUT ---
    print("\n===============================================")
    print("                 FINAL RESULTS                 ")
    print("===============================================")
    print(f"{'Dataset Phase':<15} | {'Accuracy (%)':<14} | {'Loss':<10}")
    print("-----------------------------------------------")
    print(f"{'Training':<15} | {final_train_acc:<14.3f} | {final_train_loss:<10.4f}")
    print(f"{'Validation':<15} | {final_val_acc:<14.3f} | {final_val_loss:<10.4f}")
    print(f"{'Testing':<15} | {final_test_acc:<14.3f} | {final_test_loss:<10.4f}")
    print("-----------------------------------------------")
    print(f"{'Total Training Time':<20} : {total_train_time:.4f} seconds")
    print(f"{'Throughput':<20} : {gflops:.3f} GFLOPS")
    print("===============================================")

    sc.stop()

if __name__ == "__main__":
    main()