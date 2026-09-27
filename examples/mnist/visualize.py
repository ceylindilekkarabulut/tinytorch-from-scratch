import numpy as np
import matplotlib.pyplot as plt
from tinytorch import Tensor, Conv2d, MaxPool2d, Linear, ReLU

# --- 1) Veri: train.py'deki load_mnist + on isleme, birebir ayni ---
import urllib.request, os

def load_mnist(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(filepath):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, filepath)
    with np.load(filepath) as d:
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]

x_train, y_train, x_test, y_test = load_mnist()
x_test = x_test.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
y_test = y_test.astype(np.int64)
# (x_train'i burada kullanmiyoruz, on islemesine gerek yok)

# --- 2) Model tanimi: train.py'deki CNN sinifi, birebir kopya ---
class CNN:
    ...  # train.py'den sinifin tamamini kopyala
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

# --- 3) Egitilmis agirliklari yukle ---
model = CNN()
w = np.load("model_weights.npz")
for p, k in zip(model.parameters(), w.files):
    p.data[:] = w[k]

# --- 4) Batch yardimcisi (show_predictions bunu kullaniyor) ---
def batches(x, y, batch_size=64, shuffle=True):
    idx = np.random.permutation(len(x)) if shuffle else np.arange(len(x))
    for i in range(0, len(x), batch_size):
        b = idx[i:i+batch_size]
        yield Tensor(x[b]), Tensor(y[b]), y[b]
    
def show_predictions(n=16, only_errors=False):
    # Tum test setinde tahmin al (evaluate'e benzer, ama tahminleri sakliyoruz)
    all_preds = []
    for xb, _, _ in batches(x_test, y_test, batch_size=256, shuffle=False):
        all_preds.append(model.forward(xb).data.argmax(axis=1))
    preds = np.concatenate(all_preds)

    if only_errors:
        idx = np.where(preds != y_test)[0]      # sadece hatalar
        title = "Yanlis tahminler"
    else:
        idx = np.random.permutation(len(y_test))  # rastgele karisim
        title = "Ornek tahminler"
    idx = idx[:n]

    rows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(9, 2.4 * rows))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(x_test[i, 0], cmap="gray")     # (1,28,28) -> (28,28)
        ok = preds[i] == y_test[i]
        ax.set_title(f"tahmin: {preds[i]}  (gercek: {y_test[i]})",
                     color="green" if ok else "red", fontsize=10)
        ax.axis("off")
    for ax in axes.flat[len(idx):]:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"predictions{'_errors' if only_errors else ''}.png", dpi=150)
    plt.show()

show_predictions(16)                  # rastgele 16 ornek
show_predictions(16, only_errors=True)  # modelin sasirdigi 16 ornek