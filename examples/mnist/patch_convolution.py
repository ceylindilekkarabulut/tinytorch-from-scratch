path = r"C:\Users\cdile\project\tinytorch\tinytorch\core\spatial.py"
s = open(path, encoding="utf-8").read()

if "_convolve_im2col" in s:
    print("Yama zaten uygulanmis, tekrar eklenmedi.")
    raise SystemExit

new_code = '''
    def _im2col(self, padded, out_h, out_w):
        """Kayan pencereleri sutunlara dizer: (B, C*kh*kw, out_h*out_w)"""
        batch_size = padded.shape[0]
        kernel_h, kernel_w = self.kernel_size
        s = self.stride
        cols = np.empty((batch_size, self.in_channels, kernel_h, kernel_w,
                         out_h, out_w), dtype=padded.dtype)
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                cols[:, :, kh, kw, :, :] = padded[:, :,
                                                  kh:kh + s * out_h:s,
                                                  kw:kw + s * out_w:s]
        return cols.reshape(batch_size,
                            self.in_channels * kernel_h * kernel_w,
                            out_h * out_w)

    def _convolve_im2col(self, padded, batch_size, out_h, out_w):
        """im2col + tek matris carpimi ile konvolusyon (loops ile ayni sonuc)."""
        cols = self._im2col(padded, out_h, out_w)              # (B, C*kh*kw, L)
        W = self.weight.data.reshape(self.out_channels, -1)     # (O, C*kh*kw)
        out = np.matmul(W, cols)                                # (B, O, L)
        return out.reshape(batch_size, self.out_channels, out_h, out_w)

'''

anchor = "    def forward(self, x):"
pos = s.find(anchor)                      # ilk 'def forward' = Conv2d'ninki
if pos == -1:
    print("HATA: forward bulunamadi, dosya degistirilmedi.")
    raise SystemExit

s = s[:pos] + new_code + s[pos:]
open(path, "w", encoding="utf-8").write(s)
print("Yama uygulandi. Kontrol:", "_convolve_im2col" in open(path, encoding="utf-8").read())