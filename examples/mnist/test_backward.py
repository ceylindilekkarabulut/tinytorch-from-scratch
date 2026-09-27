import numpy as np
from tinytorch import Conv2d, Tensor
from tinytorch.core.spatial import Conv2dBackward
import time

rng = np.random.default_rng(1)
for k, p, st in [(3, 1, 1), (3, 0, 1), (5, 2, 1), (3, 1, 2)]:
    conv = Conv2d(2, 4, kernel_size=k, padding=p, stride=st)
    x = Tensor(rng.standard_normal((4, 2, 14, 14)).astype(np.float32),
               requires_grad=True)
    out = conv(x)
    g = rng.standard_normal(out.shape).astype(np.float32)

    fn = Conv2dBackward(x, conv.weight, conv.bias, st, p,
                        conv.kernel_size, conv._apply_padding(x.data).shape)
    t0 = time.time(); slow = fn.apply_slow(g); t_s = time.time() - t0
    t0 = time.time(); fast = fn.apply(g);      t_f = time.time() - t0

    diffs = [np.abs(np.asarray(a) - np.asarray(b)).max()
             for a, b in zip(slow, fast)]
    print(f"k={k} p={p} s={st} | farklar: {[f'{d:.2e}' for d in diffs]} "
          f"| hizlanma: {t_s/max(t_f,1e-9):.0f}x")