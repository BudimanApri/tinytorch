import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tinytorch.seed import seed_everything
from tinytorch.nn import MLP

def main():
    # Seed everything for deterministic results
    seed_everything(1337)

    # 1. Generate two-moons dataset
    X, y = make_moons(n_samples=100, noise=0.1, random_state=1337)
    # Convert labels from {0, 1} to {-1, 1} for SVM max-margin loss
    y = y * 2 - 1

    # 2. Initialize the MLP model
    # Input dimension: 2 (x, y coordinates)
    # Hidden layers: [16, 16]
    # Output dimension: 1 (score)
    model = MLP(2, [16, 16, 1])
    print(f"Initialized MLP model with {len(model.parameters())} parameters.")

    # Training parameters
    epochs = 100
    learning_rate = 1.0

    # 3. Training Loop
    for epoch in range(epochs):
        # Prepare inputs as lists of Value nodes
        # X is (100, 2), we convert each sample into a list of two Values
        
        # TODO: Implement the 5 canonical training steps:
        
        # Step 1: zero_grad
        # Reset the gradients of all parameters to 0.0
        zero_grad = model.zero_grad()
        
        # Step 2: forward
        # Run each input through the model. Since Layer/MLP expects input list/sequence,
        # you can pass list(x) to model. For example:
        # scores = [model(list(x_sample)) for x_sample in X]
        score = []
        for i in X:
            score.append(model(list(i)))
        
        # Step 3: loss
        # Compute SVM max-margin loss: mean of relu(1.0 - y_i * score_i)
        # Plus L2 regularization (reg_loss = 1e-4 * sum(p*p for p in model.parameters()))
        # Hint: use `(1.0 - y_i * score_i).relu()` to compute the margin loss for each sample
        margin_loss = []
        for y_i, score_i in zip(y, score):
            margin_loss.append((1.0 - y_i* score_i).relu())
        reg_loss = 1e-4 * sum(p*p for p in model.parameters())
        margin_loss = sum(margin_loss)/len(margin_loss)
        total_loss = margin_loss + reg_loss

        # Step 4: backward
        # Call backward() on the total loss
        total_loss.backward()
        
        # Step 5: step (SGD)
        # Update each parameter's data: p.data -= learning_rate * p.grad
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        # --- end of TODO ---

        # Compute accuracy (scores are Value objects)
        predictions = [1.0 if s.data > 0 else -1.0 for s in score]
        accuracy = sum(1.0 for p, yi in zip(predictions, y) if p == yi) / len(y)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {total_loss.data:.4f} | Accuracy: {accuracy*100:.1f}%")

    # 4. Plot Decision Boundary
    print("Training complete! Plotting decision boundary...")
    os.makedirs("assets", exist_ok=True)
    
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Forward pass on the grid points to generate predictions
    Z = []
    for pt in grid_points:
        score = model([float(pt[0]), float(pt[1])])
        Z.append(score.data)
    Z = np.array(Z).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Two Moons Decision Boundary (tinytorch)")
    
    output_img = "assets/moons_decision_boundary.png"
    plt.savefig(output_img)
    plt.close()
    print(f"Decision boundary saved as {output_img}")

if __name__ == "__main__":
    main()
