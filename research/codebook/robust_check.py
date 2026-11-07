import sys; sys.path.insert(0, "/src/research/codebook")
import torch, viterbi_sim as vs
def cb(c):
    x = (vs.S * c) & 0xFFFFFFFF
    return vs.norm(sum(vs.bytes_of(x, 4)).float())
torch.manual_seed(0)
w = torch.randn(32, 512, device=vs.DEV)
for K in (3, 6):
    rd = 2.0 ** (-2 * K)
    al = [0.98, 1.02, 1.06] if K == 3 else [1.0, 1.04, 1.08]
    base = vs.best_mse(vs.norm(vs.cb_mul1()), K, w, al)[0]
    print(f"K={K} mul1 {base:.5f} ({base/rd:.3f} x bound)")
    for c in (0x7FD633, 0x7CE42D, 0xDCD12D):
        m = vs.best_mse(cb(c), K, w, al)[0]
        print(f"   C=0x{c:06X}: {m:.5f} ({m/base:.3f} x mul1)", flush=True)
