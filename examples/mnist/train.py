import numpy as np
from tinytorch import Tensor, Conv2d, MaxPool2d, Linear, ReLU, CrossEntropyLoss, Adam
import urllib.request
import os

def load_mnist(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(filepath):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print("Downloading MNIST...")
        urllib.request.urlretrieve(url, filepath)
        print("Done.")
    with np.load(filepath) as d:
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]

x_train, y_train, x_test, y_test = load_mnist()

#print("x_train:", x_train.shape, x_train.dtype)
#print("y_train:", y_train.shape, y_train.dtype)
#print("x_test:", x_test.shape)
#print("y_test:", y_test.shape)

x_train = x_train.astype(np.float32) / 255.0
x_test  = x_test.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 1, 28, 28)
x_test  = x_test.reshape(-1, 1, 28, 28)

y_train = y_train.astype(np.int64)
y_test  = y_test.astype(np.int64)

#print("x_train:", x_train.shape, x_train.dtype, "| min:", x_train.min(), "max:", x_train.max())
#print("x_test: ", x_test.shape, x_test.dtype)
#print("y_train:", y_train.shape, y_train.dtype, "| örnek etiketler:", y_train[:10])

class CNN:
    def __init__(self):
        self.conv1 = Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(2)
        self.pool2 = MaxPool2d(2)
        self.relu = ReLU()
        self.fc1 = Linear(16 * 7 * 7, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))   
        x = self.pool2(self.relu(self.conv2(x)))   
        x = x.reshape(x.shape[0], -1)               
        x = self.relu(self.fc1(x))                  
        return self.fc2(x)                          

    def parameters(self):
        params = []
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
            params += list(layer.parameters())
        return params


model = CNN()
x_try = Tensor(x_train[:4])          
out = model.forward(x_try)
#print("Shape testi -> girdi:", x_try.shape, "| çıktı:", out.shape)
#print("Parametre grubu sayısı:", len(model.parameters()))
# --- Hiz + saglik testi (yamali conv) ---

import time

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def batches(x, y, batch_size=64, shuffle=True):
    idx = np.random.permutation(len(x)) if shuffle else np.arange(len(x))
    for i in range(0, len(x), batch_size):
        b = idx[i:i+batch_size]
        yield Tensor(x[b]), Tensor(y[b]), y[b]

def evaluate(x, y, batch_size=256):
    correct = 0
    for xb, _, y_np in batches(x, y, batch_size, shuffle=False):
        preds = model.forward(xb).data.argmax(axis=1)
        correct += (preds == y_np).sum()
    return correct / len(x)

# Ilk tur: 10.000 ornekle hiz olcumu + dogrulama
N_TRAIN = 60_000
EPOCHS = 5

for epoch in range(EPOCHS):
    t0 = time.time()
    epoch_loss, n_batches = 0.0, 0
    for xb, yb, _ in batches(x_train[:N_TRAIN], y_train[:N_TRAIN]):
        logits = model.forward(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.data)
        n_batches += 1
    test_acc = evaluate(x_test, y_test)
    print(f"Epoch {epoch+1}/{EPOCHS} | loss: {epoch_loss/n_batches:.4f} "
          f"| test acc: {test_acc:.2%} | sure: {time.time()-t0:.1f}s")

np.savez("model_weights.npz", *[p.data for p in model.parameters()])
print("Agirliklar kaydedildi: model_weights.npz")