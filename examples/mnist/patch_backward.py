path = r"C:\Users\cdile\project\tinytorch\tinytorch\core\spatial.py"
s = open(path, encoding="utf-8").read()

if "def apply_slow" in s:
    print("Yama zaten uygulanmis."); raise SystemExit

fast = '''    def apply(self, grad_output):
        """Vektorize backward: im2col + matmul (apply_slow ile ayni sonuc)."""
        grad_output = np.asarray(grad_output)
        batch_size, out_channels, out_h, out_w = grad_output.shape
        _, in_channels, in_height, in_width = self.x.shape
        kernel_h, kernel_w = self.kernel_size
        st = self.stride

        if self.padding > 0:
            padded_input = np.pad(self.x.data,
                ((0, 0), (0, 0), (self.padding, self.padding),
                 (self.padding, self.padding)),
                mode='constant', constant_values=0)
        else:
            padded_input = np.asarray(self.x.data)

        # girdiyi sutunlara diz (forward'daki _im2col ile ayni mantik)
        cols = np.empty((batch_size, in_channels, kernel_h, kernel_w,
                         out_h, out_w), dtype=padded_input.dtype)
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                cols[:, :, kh, kw, :, :] = padded_input[:, :,
                    kh:kh + st * out_h:st, kw:kw + st * out_w:st]
        cols = cols.reshape(batch_size, in_channels * kernel_h * kernel_w, -1)

        g = grad_output.reshape(batch_size, out_channels, -1)   # (B, O, L)

        # dW = sum_b g @ cols^T
        grad_weight = np.matmul(g, cols.transpose(0, 2, 1)).sum(axis=0)
        grad_weight = grad_weight.reshape(self.weight.data.shape)

        # dcols = W^T @ g, sonra col2im ile geri sac
        W = self.weight.data.reshape(out_channels, -1)
        gcols = np.matmul(W.T, g).reshape(batch_size, in_channels,
                                          kernel_h, kernel_w, out_h, out_w)
        grad_input_padded = np.zeros_like(padded_input)
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                grad_input_padded[:, :,
                    kh:kh + st * out_h:st,
                    kw:kw + st * out_w:st] += gcols[:, :, kh, kw, :, :]

        if self.padding > 0:
            grad_input = grad_input_padded[:, :,
                self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        grad_bias = None if self.bias is None else grad_output.sum(axis=(0, 2, 3))

        if self.bias is None:
            return grad_input, grad_weight
        return grad_input, grad_weight, grad_bias

'''

anchor = "    def apply(self, grad_output):"
pos = s.find(anchor)                       # ilki = Conv2dBackward'inki
if pos == -1:
    print("HATA: apply bulunamadi."); raise SystemExit

s = s[:pos] + fast + s[pos:].replace(anchor, "    def apply_slow(self, grad_output):", 1)
open(path, "w", encoding="utf-8").write(s)
print("Backward yamalandi:", "apply_slow" in open(path, encoding="utf-8").read())