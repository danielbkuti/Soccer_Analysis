import numpy as np

np.random.seed(42)

x = np.linspace(0, 10, 100)
y = 3 * x + 7 + 1.5 * np.random.randn(100)

print(x[:5])
print(y[:5])

w = 0.0
b = 0.0

learning_rate = 0.01

loss_history = []

for epoch in range(1000):
    y_pred = w * x + b

    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)

    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(loss_history, color='steelblue')
ax1.set_title("Loss Curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")

ax2.scatter(x, y, color='steelblue', alpha=0.5, label='Actual data')
ax2.plot(x, w * x + b, color='red', label=f'Learned: y = {w:.2f}x + {b:.2f}')
ax2.set_title("Fitted Line vs Data")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend()

plt.tight_layout()
plt.show()