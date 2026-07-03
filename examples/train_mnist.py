import os
import time
import urllib.request
import gzip
import numpy as np
from tinytorch.engine import Tensor
from tinytorch.nn import Sequential, Linear, ReLU
from tinytorch.optim import Adam
from tinytorch.functional import cross_entropy

def download_mnist(target_dir="data"):
    """Downloads MNIST files if they are not already present."""
    os.makedirs(target_dir, exist_ok=True)
    urls = {
        "train_img": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "train_lbl": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "test_img": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "test_lbl": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
    }
    
    paths = {}
    for name, url in urls.items():
        filename = url.split("/")[-1]
        filepath = os.path.join(target_dir, filename)
        paths[name] = filepath
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            # Set a User-Agent to avoid HTTP 403 Forbidden from some mirrors
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, filepath)
            
    return paths

def load_mnist(paths):
    """Loads and normalizes MNIST images and labels."""
    def read_images(path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 28 * 28).astype(np.float64) / 255.0

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data

    x_train = read_images(paths["train_img"])
    y_train = read_labels(paths["train_lbl"])
    x_test = read_images(paths["test_img"])
    y_test = read_labels(paths["test_lbl"])
    
    return x_train, y_train, x_test, y_test

def to_one_hot(y, num_classes=10):
    """Converts a label vector to a one-hot matrix."""
    oh = np.zeros((len(y), num_classes), dtype=np.float64)
    oh[np.arange(len(y)), y] = 1.0
    return oh

def main():
    # Load dataset
    paths = download_mnist()
    x_train, y_train, x_test, y_test = load_mnist(paths)
    
    # We subset to 1000 train samples and 200 test samples to keep the training
    # extremely fast (under 10 seconds) on CPU in pure python.
    num_train = 1000
    num_test = 200
    x_train, y_train = x_train[:num_train], y_train[:num_train]
    x_test, y_test = x_test[:num_test], y_test[:num_test]
    
    # Convert labels to one-hot for our cross_entropy loss
    y_train_oh = to_one_hot(y_train)
    y_test_oh = to_one_hot(y_test)
    
    # Initialize the MLP model
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10)
    )
    print(f"Initialized tinytorch MLP with {len(model.parameters())} parameters.")
    
    # Initialize Adam optimizer
    optimizer = Adam(model.parameters(), lr=0.01)
    
    batch_size = 100
    epochs = 10
    
    print("=(Bismillah Work!!)= tinytorch MNIST Training ===")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.arange(num_train)
        np.random.shuffle(indices)
        x_train_shuffled = x_train[indices]
        y_train_oh_shuffled = y_train_oh[indices]
        
        epoch_loss = 0.0
        
        # Mini-batch training
        for i in range(0, num_train, batch_size):
            # 1. Slice mini-batch data
            bx = x_train_shuffled[i : i + batch_size]
            by = y_train_oh_shuffled[i : i + batch_size]
            
            # Wrap in Tensor objects
            batch_x = Tensor(bx)
            batch_y = Tensor(by)
            
            # TODO: Implement the training step:
            # 1. Zero the parameter gradients via the optimizer
            optimizer.zero_grad()
            # 2. Forward pass: compute predictions by calling model(batch_x)
            predictions = model(batch_x)
            # 3. Loss: compute cross_entropy loss between predictions and batch_y
            loss = cross_entropy(predictions, batch_y)
            # 4. Backward pass: backpropagate gradients
            loss.backward()
            # 5. Optimization step: update weights via the optimizer
            optimizer.step()
            # 6. Accumulate loss: epoch_loss += loss.data.item() * (len(bx) / num_train)
            epoch_loss += loss.data * (len(bx) / num_train)
            
        # Calculate test accuracy
        # Forward pass on test set (no tracking gradients needed since we don't call backward)
        test_x = Tensor(x_test)
        test_preds = model(test_x)
        pred_labels = np.argmax(test_preds.data, axis=1)
        accuracy = np.mean(pred_labels == y_test)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss:.4f} | Test Acc: {accuracy*100:.1f}%")
        
    elapsed = time.time() - start_time
    print(f"tinytorch training completed in {elapsed:.4f} seconds. Yeyyy")

if __name__ == "__main__":
    main()
