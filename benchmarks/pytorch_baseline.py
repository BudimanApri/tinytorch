import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from examples.train_mnist import download_mnist, load_mnist
from tinytorch.seed import SEED

def main():
    torch.manual_seed(SEED)
    
    # Load dataset using same loader
    paths = download_mnist()
    x_train, y_train, x_test, y_test = load_mnist(paths)
    
    # Subset to match tinytorch dataset size for apples-to-apples comparison
    num_train = 1000
    num_test = 200
    x_train, y_train = x_train[:num_train], y_train[:num_train]
    x_test, y_test = x_test[:num_test], y_test[:num_test]
    
    # Convert to torch tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # Define model identical to tinytorch structure
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Same architecture and hyperparameters as tinytorch; weights are NOT copied —
    # each library uses its own default initialization scheme.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    batch_size = 100
    epochs = 10
    
    print("=== PyTorch MNIST Baseline ===")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(x_train_t.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, x_train_t.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train_t[indices], y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (batch_x.size(0) / num_train)
            
        # Test accuracy
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test_t)
            predictions = torch.argmax(test_outputs, dim=1)
            accuracy = (predictions == y_test_t).float().mean().item()
            
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss:.4f} | Test Acc: {accuracy*100:.1f}%")
        
    elapsed = time.time() - start_time
    print(f"PyTorch training completed in {elapsed:.4f} seconds.")

if __name__ == "__main__":
    main()
