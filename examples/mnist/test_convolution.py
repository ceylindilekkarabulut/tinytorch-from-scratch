import numpy as np
from tinytorch import Conv2d, Tensor
import time

rng = np.random.default_rng(0)
x = rng.standard_normal((8, 1, 28, 28)).astype(np.float32)

conv = Conv2d(1, 8, kernel_size=3, padding=1)
padded = conv._apply_padding(x)
oh, ow = conv._compute_output_shape(28, 28)

t0 = time.time(); slow = conv._convolve_loops(padded, 8, oh, ow); t_slow = time.time() - t0
t0 = time.time(); fast = conv._convolve_im2col(padded, 8, oh, ow); t_fast = time.time() - t0

print("max fark:", np.abs(slow - fast).max())
print(f"yavas: {t_slow:.3f}s | hizli: {t_fast:.4f}s | hizlanma: {t_slow/max(t_fast,1e-9):.0f}x")

# stride/padding varyasyonlari da dogru mu?
for k, p, s in [(3,0,1), (5,2,1), (3,1,2)]:
    c = Conv2d(2, 4, kernel_size=k, padding=p, stride=s)
    xi = rng.standard_normal((2, 2, 15, 15))
    pd = c._apply_padding(xi)
    o_h, o_w = c._compute_output_shape(15, 15)
    d = np.abs(c._convolve_loops(pd, 2, o_h, o_w) - c._convolve_im2col(pd, 2, o_h, o_w)).max()
    print(f"k={k} p={p} s={s} -> max fark: {d:.2e}")