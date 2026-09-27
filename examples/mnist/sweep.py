import numpy as np
import time, urllib.request, os
from tinytorch import Tensor, Conv2d, MaxPool2d, Linear, ReLU, CrossEntropyLoss, Adam

# --- veri (train.py'dekiyle ayni) ---
def load_mnist(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    fp = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(fp):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", fp)
    with np.load(fp) as d:
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]

x_train, y_train, x_test, y_test = load_mnist()
x_train = x_train.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
x_test  = x_test.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
y_train = y_train.astype(np.int64); y_test = y_test.astype(np.int64)

# --- parametrik model ---
class CNN:
    def __init__(self, ch1=8, ch2=16, hidden=128, kernel=3):
        pad = kernel // 2
        self.conv1 = Conv2d(1, ch1, kernel_size=kernel, padding=pad)
        self.conv2 = Conv2d(ch1, ch2, kernel_size=kernel, padding=pad)
        self.pool1, self.pool2 = MaxPool2d(2), MaxPool2d(2)
        self.relu = ReLU()
        self.fc1 = Linear(ch2 * 7 * 7, hidden)
        self.fc2 = Linear(hidden, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        return self.fc2(self.relu(self.fc1(x)))

    def parameters(self):
        p = []
        for l in [self.conv1, self.conv2, self.fc1, self.fc2]:
            p += list(l.parameters())
        return p

def batches(x, y, bs, shuffle=True):
    idx = np.random.permutation(len(x)) if shuffle else np.arange(len(x))
    for i in range(0, len(x), bs):
        b = idx[i:i+bs]
        yield Tensor(x[b]), Tensor(y[b]), y[b]

def run_experiment(name, lr=0.001, bs=64, ch1=8, ch2=16, hidden=128,
                   kernel=3, n_train=10_000, epochs=2, seed=0):
    np.random.seed(seed)
    model = CNN(ch1, ch2, hidden, kernel)
    crit, opt = CrossEntropyLoss(), Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        for xb, yb, _ in batches(x_train[:n_train], y_train[:n_train], bs):
            loss = crit(model.forward(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    correct = 0
    for xb, _, ynp in batches(x_test, y_test, 256, shuffle=False):
        correct += (model.forward(xb).data.argmax(axis=1) == ynp).sum()
    acc, dt = correct / len(y_test), time.time() - t0
    print(f"{name:28s} | acc: {acc:.2%} | sure: {dt:.0f}s")
    return name, acc, dt

# --- experiment plan: baseline + one-factor-at-a-time changes ---
results = []
results.append(run_experiment("BASELINE (lr.001 bs64 8/16/128)"))
results.append(run_experiment("lr = 0.01",   lr=0.01))
results.append(run_experiment("lr = 0.0001", lr=0.0001))
results.append(run_experiment("batch = 16",  bs=16))
results.append(run_experiment("batch = 256", bs=256))
results.append(run_experiment("channels 4/8",   ch1=4,  ch2=8))
results.append(run_experiment("channels 16/32", ch1=16, ch2=32))
results.append(run_experiment("hidden = 32",  hidden=32))
results.append(run_experiment("hidden = 256", hidden=256))
results.append(run_experiment("kernel = 5", kernel=5))

# --- report ---
with open("sweep_report.md", "w", encoding="utf-8") as f:
    f.write("# Hyperparameter Sweep (10k samples, 2 epochs)\n\n")
    f.write("| Experiment | Test Accuracy | Time (s) |\n|---|---|---|\n")
    base_acc = results[0][1]
    for name, acc, dt in results:
        delta = f" ({acc-base_acc:+.2%})" if name != results[0][0] else ""
        f.write(f"| {name} | {acc:.2%}{delta} | {dt:.0f} |\n")
print("\nReport written: sweep_report.md")