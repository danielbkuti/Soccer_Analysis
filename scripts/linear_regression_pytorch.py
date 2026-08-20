import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

np.random.seed(42)
x_np = np.linspace(0, 10, 100)
y_np = 3 * x_np + 7 + 1.5 * np.random.randn(100)

x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

print(x.shape)
print(y.shape)

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01
optimizer = optim.SGD([w, b], lr=learning_rate)

loss_history = []

for epoch in range(1000):
    y_pred = w * x + b

    loss = torch.mean((y_pred - y) ** 2)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | w: {w.item():.4f} | b: {b.item():.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history, color='steelblue')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.subplot(1, 2, 2)
plt.scatter(x.detach(), y.detach(), color='steelblue', alpha=0.5, label='Actual data')
plt.plot(x.detach(), (w * x + b).detach(), color='red', label=f'Learned: y = {w.item():.2f}x + {b.item():.2f}')
plt.title("Fitted Line vs Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()