// Decode-cost microbenchmark for INT8-WMMA GEMV with an on-the-fly trellis decode (gfx1201).
// Only the speed of the decode path is measured (random trellis data); correctness of each variant's
// format is a separate question (see viterbi_sim.py for accuracy).
//
//   V0  tile layout: tensor-core permutation, decoded values go through LDS (shelf's current kernel)
//   V1  WMMA-B-fragment layout (lane = column n, 8 consecutive k), mul1 (v_mul_lo_u32) + dot4 + >>2
//   V2  fragment layout, 24-bit-constant multiply (full rate) + dot4 + >>2
//   V3  fragment layout, 24-bit multiply, 6-bit fields: sad(x & 0x3F3F3F3F)  (no shift)
//   V4  fragment layout, two table lookups through LDS (2 x 256-entry byte tables)
//   V5  fragment layout, 4 x 16-entry nibble tables through v_perm (K=4 stream nibbles)
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

constexpr int BITS = 4;
constexpr uint32_t kWords = 8u * BITS;
constexpr uint32_t kBitsTotal = 256u * BITS;

template <int V>
__device__ __forceinline__ uint32_t weight_byte(uint32_t state, const uint8_t* lds_tab) {
    if constexpr (V == 1) {
        return __builtin_amdgcn_udot4(state * 0x83DCD12Du, 0x01010101u, 3u, false) >> 2u;
    } else if constexpr (V == 2) {
        return __builtin_amdgcn_udot4(__umul24(state, 0xDCD12Du), 0x01010101u, 3u, false) >> 2u;
    } else if constexpr (V == 3) {
        return __builtin_amdgcn_sad_u8(__umul24(state, 0xDCD12Du) & 0x3F3F3F3Fu, 0u, 0u);
    } else if constexpr (V == 4) {
        return (uint32_t)lds_tab[state & 255u] + (uint32_t)lds_tab[256u + (state >> 8u)];
    } else {
        return state;
    }
}

// Fragment layout kernel: each lane decodes its own 8 consecutive stream positions into one WMMA B fragment.
template <int V, int MT>
__global__ void __launch_bounds__(128) gemv_frag(
    const int8_t* __restrict__ a, const uint32_t* __restrict__ trellis, int32_t* __restrict__ sink,
    uint32_t n, uint32_t k) {
    __shared__ uint8_t tab[512];
    if (V == 4) for (uint32_t i = threadIdx.x; i < 512; i += 128) tab[i] = (uint8_t)(i * 37u);
    __syncthreads();
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
    const uint32_t nt = blockIdx.x * 4u + wave;
    const uint32_t kt_count = k >> 4u;
    const uint32_t base_bit = (lane * 8u * BITS + BITS + kBitsTotal - 16u) % kBitsTotal;
    const uint32_t off = base_bit & 31u;
    const uint32_t i0 = base_bit >> 5u, i1 = (i0 + 1u) % kWords, i2 = (i0 + 2u) % kWords;
    const uint32_t* tile_base = trellis + (size_t)nt * kt_count * kWords;
    const int8_t* arow = a + (lane & 15u) * k + (lane >> 4u) * 8u;
    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t groups = k >> 7u;
    uint32_t cur[8][3], nxt[8][3];
    #pragma unroll
    for (uint32_t s = 0; s < 8; ++s) { const uint32_t* t = tile_base + (size_t)s * kWords; cur[s][0] = t[i0]; cur[s][1] = t[i1]; cur[s][2] = t[i2]; }
    for (uint32_t g = 0; g < groups; ++g) {
        if (g + 1u < groups) {
            #pragma unroll
            for (uint32_t s = 0; s < 8; ++s) { const uint32_t* t = tile_base + (size_t)((g + 1u) * 8u + s) * kWords; nxt[s][0] = t[i0]; nxt[s][1] = t[i1]; nxt[s][2] = t[i2]; }
        }
        #pragma unroll
        for (uint32_t s = 0; s < 8; ++s) {
            uint32_t packed[2] = {0u, 0u};
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                constexpr uint32_t off_c = (BITS + 32u - 16u) % 32u;
                const uint32_t pos = off_c + i * BITS, e = pos + 16u, wl = (e - 1u) >> 5u, r = 32u - (e - 32u * wl);
                const uint32_t lo = wl == 0u ? cur[s][0] : (wl == 1u ? cur[s][1] : cur[s][2]);
                const uint32_t hi = wl == 0u ? 0u : (wl == 1u ? cur[s][0] : cur[s][1]);
                const uint32_t state = __builtin_amdgcn_alignbit(hi, lo, r) & 0xFFFFu;
                packed[i >> 2u] |= (weight_byte<V>(state, tab) & 0xFFu) << ((i & 3u) * 8u);
            }
            int32x2_t b = {(int32_t)packed[0], (int32_t)packed[1]};
            int32x2_t av = *reinterpret_cast<const int32x2_t*>(arow + g * 128u + s * 16u);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av, false, b, acc, false);
        }
        #pragma unroll
        for (uint32_t s = 0; s < 8; ++s) { cur[s][0] = nxt[s][0]; cur[s][1] = nxt[s][1]; cur[s][2] = nxt[s][2]; }
    }
    int32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) sum += acc[i];
    if (sum == 0x7fffffff) sink[0] = sum;
}

// Tile layout (current shelf kernel): values go through LDS to be re-read as B fragments.
__global__ void __launch_bounds__(128) gemv_tile_lds(
    const int8_t* __restrict__ a, const uint32_t* __restrict__ trellis, int32_t* __restrict__ sink,
    uint32_t n, uint32_t k) {
    __shared__ __attribute__((aligned(16))) int8_t lds[4][8][256];
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
    const uint32_t nt = blockIdx.x * 4u + wave;
    const uint32_t kt_count = k >> 4u;
    const uint32_t base_bit = (lane * 8u * BITS + BITS + kBitsTotal - 16u) % kBitsTotal;
    const uint32_t i0 = base_bit >> 5u, i1 = (i0 + 1u) % kWords, i2 = (i0 + 2u) % kWords;
    const uint32_t wa = (lane & 3u) << 2u, wc = lane >> 2u, col = lane & 15u, half = lane >> 4u;
    const uint32_t* tile_base = trellis + (size_t)nt * kt_count * kWords;
    const int8_t* arow = a + col * k + half * 8u;
    int32x8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t groups = k >> 7u;
    uint32_t cur[8][3], nxt[8][3];
    #pragma unroll
    for (uint32_t s = 0; s < 8; ++s) { const uint32_t* t = tile_base + (size_t)s * kWords; cur[s][0] = t[i0]; cur[s][1] = t[i1]; cur[s][2] = t[i2]; }
    for (uint32_t g = 0; g < groups; ++g) {
        if (g + 1u < groups) {
            #pragma unroll
            for (uint32_t s = 0; s < 8; ++s) { const uint32_t* t = tile_base + (size_t)((g + 1u) * 8u + s) * kWords; nxt[s][0] = t[i0]; nxt[s][1] = t[i1]; nxt[s][2] = t[i2]; }
        }
        #pragma unroll
        for (uint32_t s = 0; s < 8; ++s) {
            uint32_t packed[2] = {0u, 0u};
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                constexpr uint32_t off_c = (BITS + 32u - 16u) % 32u;
                const uint32_t pos = off_c + i * BITS, e = pos + 16u, wl = (e - 1u) >> 5u, r = 32u - (e - 32u * wl);
                const uint32_t lo = wl == 0u ? cur[s][0] : (wl == 1u ? cur[s][1] : cur[s][2]);
                const uint32_t hi = wl == 0u ? 0u : (wl == 1u ? cur[s][0] : cur[s][1]);
                const uint32_t state = __builtin_amdgcn_alignbit(hi, lo, r) & 0xFFFFu;
                packed[i >> 2u] |= (weight_byte<1>(state, nullptr) & 0xFFu) << ((i & 3u) * 8u);
            }
            *reinterpret_cast<uint32_t*>(&lds[wave][s][wc * 16u + wa]) = packed[0] ^ 0x80808080u;
            *reinterpret_cast<uint32_t*>(&lds[wave][s][(wc + 8u) * 16u + wa]) = packed[1] ^ 0x80808080u;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        #pragma unroll
        for (uint32_t s = 0; s < 8; ++s) {
            int32x2_t b = *reinterpret_cast<const int32x2_t*>(&lds[wave][s][col * 16u + half * 8u]);
            int32x2_t av = *reinterpret_cast<const int32x2_t*>(arow + g * 128u + s * 16u);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, av, true, b, acc, false);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront"); __builtin_amdgcn_wave_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        #pragma unroll
        for (uint32_t s = 0; s < 8; ++s) { cur[s][0] = nxt[s][0]; cur[s][1] = nxt[s][1]; cur[s][2] = nxt[s][2]; }
    }
    int32_t sum = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) sum += acc[i];
    if (sum == 0x7fffffff) sink[0] = sum;
}

template <typename F>
double bench(F launch, double bytes) {
    hipEvent_t a, b; (void)hipEventCreate(&a); (void)hipEventCreate(&b);
    for (int i = 0; i < 10; ++i) launch();
    (void)hipEventRecord(a);
    const int iters = 100;
    for (int i = 0; i < iters; ++i) launch();
    (void)hipEventRecord(b); (void)hipEventSynchronize(b);
    float ms = 0; (void)hipEventElapsedTime(&ms, a, b); ms /= iters;
    return bytes / ms / 1e6;  // GB/s
}

int main(int argc, char** argv) {
    // Weights are larger than the 64 MB last-level cache so the numbers are DRAM-bound where possible.
    const uint32_t n = argc > 1 ? atoi(argv[1]) : 17408 * 2, k = argc > 2 ? atoi(argv[2]) : 5120;
    const size_t tiles = (size_t)(n / 16) * (k / 16);
    const size_t bytes = tiles * 32u * BITS;
    uint32_t* trellis; int8_t* a; int32_t* sink;
    (void)hipMalloc(&trellis, bytes); (void)hipMalloc(&a, 16u * k); (void)hipMalloc(&sink, 64);
    std::vector<uint32_t> h(bytes / 4);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (uint32_t)(i * 2654435761u + 12345u);
    (void)hipMemcpy(trellis, h.data(), bytes, hipMemcpyHostToDevice);
    (void)hipMemset(a, 1, 16u * k);
    const dim3 grid(n / 64u), block(128);
    printf("N=%u K=%u bits=%d, weights %.1f MB\n", n, k, BITS, bytes / 1e6);
    printf("V0 tile layout + LDS (mul1)         : %6.1f GB/s\n", bench([&] { gemv_tile_lds<<<grid, block>>>(a, trellis, sink, n, k); }, (double)bytes));
    printf("V1 frag layout, mul1 + dot4 + >>2   : %6.1f GB/s\n", bench([&] { gemv_frag<1, 1><<<grid, block>>>(a, trellis, sink, n, k); }, (double)bytes));
    printf("V2 frag layout, mul24 + dot4 + >>2  : %6.1f GB/s\n", bench([&] { gemv_frag<2, 1><<<grid, block>>>(a, trellis, sink, n, k); }, (double)bytes));
    printf("V3 frag layout, mul24 + and + sad   : %6.1f GB/s\n", bench([&] { gemv_frag<3, 1><<<grid, block>>>(a, trellis, sink, n, k); }, (double)bytes));
    printf("V4 frag layout, 2 byte tables (LDS) : %6.1f GB/s\n", bench([&] { gemv_frag<4, 1><<<grid, block>>>(a, trellis, sink, n, k); }, (double)bytes));
    printf("V9 frag layout, no decode (ceiling) : %6.1f GB/s\n", bench([&] { gemv_frag<9, 1><<<grid, block>>>(a, trellis, sink, n, k); }, (double)bytes));
    return 0;
}
