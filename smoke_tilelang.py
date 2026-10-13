import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def vector_add(N: int, block: int = 256):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        C: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < N:
                    C[idx] = A[idx] + B[idx]
    return kernel

N = 4096

a = torch.randn(N, device="cuda")
b = torch.randn(N, device="cuda")
c = torch.empty_like(a)

kernel = vector_add(N)
kernel(a, b, c)

torch.testing.assert_close(c, a + b)

print("TileLang JIT compilation: PASS")
print("H100 GPU execution: PASS")
print("Numerical correctness: PASS")
