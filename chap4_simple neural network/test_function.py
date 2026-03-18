import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ===== 函数 =====
def target_function(x):
    return np.sin(np.sqrt(x)) - 2 * np.log(x)


# ===== 数据 =====
def generate_data(n=1000):
    x = np.linspace(0.1, 10, n)
    y = target_function(x)

    idx = np.random.permutation(n)
    x = x[idx]
    y = y[idx]

    return x.reshape(-1, 1), y.reshape(-1, 1)


x, y = generate_data()

# 划分
split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]


# ===== 标准化 =====
mean = x_train.mean()
std = x_train.std()

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


# ===== tensor =====
x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)


# ===== 两层 ReLU 网络 =====
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.relu = nn.ReLU()

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x


model = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


# ===== 训练 =====
loss_list = []

for epoch in range(3000):
    model.train()

    pred = model(x_train_t)
    loss = criterion(pred, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f}")


# ===== 测试 =====
model.eval()
with torch.no_grad():
    pred_test = model(x_test_t)

    test_loss = criterion(pred_test, y_test_t).item()

    # 转 numpy
    y_true = y_test_t.numpy()
    y_pred = pred_test.numpy()

    # MAE
    mae = np.mean(np.abs(y_pred - y_true))

    # RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    # R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

print("\n===== Test Metrics =====")
print(f"Test MSE  : {test_loss:.6f}")
print(f"Test MAE  : {mae:.6f}")
print(f"Test RMSE : {rmse:.6f}")
print(f"Test R^2  : {r2:.6f}")


# ===== 随机抽样对比 =====
print("\n===== Sample Predictions =====")
for i in range(10):
    print(f"x = {x_test[i][0]:.4f} | true = {y_test[i][0]:.4f} | pred = {y_pred[i][0]:.4f}")


# ===== 可视化=====
plt.figure()
plt.scatter(x_test, y_test, label='True', s=10)
plt.scatter(x_test, pred_test.detach().numpy(), label='Pred', s=10)
plt.legend()
plt.title("Test Fitting (ReLU)")
plt.show()


# ===== 整体拟合 =====
x_all = np.linspace(0.1, 10, 1000).reshape(-1, 1)
x_all_norm = (x_all - mean) / std
x_all_t = torch.tensor(x_all_norm, dtype=torch.float32)

with torch.no_grad():
    pred_all = model(x_all_t).numpy()

plt.figure()
plt.plot(x_all, target_function(x_all), label="True")
plt.plot(x_all, pred_all, label="Pred")
plt.legend()
plt.title("Overall Fitting (ReLU)")
plt.show()