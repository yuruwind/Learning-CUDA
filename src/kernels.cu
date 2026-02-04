#include <vector>
#include <cuda_fp16.h>
#include <float.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */

// 解决原子加法的类型兼容性问题
template <typename T>
__device__ inline void gpuAtomicAdd(T* address, T val);

// int specialization
template <>
__device__ inline void gpuAtomicAdd<int>(int* address, int val) {
    atomicAdd(address, val);
}

// float specialization
template <>
__device__ inline void gpuAtomicAdd<float>(float* address, float val) {
    atomicAdd(address, val);
}

template <typename T>
__global__ void calculate_trace_kernel(
    const T* __restrict__ d_input,
    T* __restrict__ d_result,
    size_t n,
    size_t cols
) {
    T local_sum = 0;

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // 规约与 Grid-Stride
    for (size_t i = tid; i < n; i += stride) {
        local_sum += d_input[i * cols + i];
    }

    // 原子操作叠加
    if (local_sum != T(0)) {
        gpuAtomicAdd<T>(d_result, local_sum);
    }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    size_t n = min(rows, cols);
    if (n == 0) return T(0);

    T *d_input = nullptr, *d_result = nullptr;
    size_t input_size = rows * cols * sizeof(T);

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_result, sizeof(T));

    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(T));

    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((n + threadsPerBlock - 1) / threadsPerBlock);

    // 限制block启动数量
    blocksPerGrid = min(blocksPerGrid, 1024);

    calculate_trace_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_result, n, cols
    );

    T h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_result);

    return h_result;
}


/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// 使用寄存器缓存 Q 和累加器，避免反复读写全局内存。
constexpr int MAX_HEAD_DIM = 128;

template <typename T>
__global__ void flash_attention_kernel_optimized(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V, T* __restrict__ O,
    int batch_size, int seq_len_q, int seq_len_kv,
    int query_heads, int kv_heads, int head_dim,
    float scale, bool is_causal
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * seq_len_q * query_heads;
    if (idx >= total_threads) return;

    // 坐标拆解 [Batch, Seq_Q, Head]
    int head_idx  = idx % query_heads;
    int q_pos     = (idx / query_heads) % seq_len_q;
    int batch_idx = idx / (query_heads * seq_len_q);

    int kv_head_idx = head_idx * kv_heads / query_heads; // GQA 处理

    // 指针偏移计算
    // Q 指向当前线程负责的那个 query 向量的起始位置
    const T* q_ptr = Q + (
        (batch_idx * seq_len_q + q_pos) * query_heads + head_idx
    ) * head_dim;

    // K, V 指向当前 batch 和 head 的起始位置 (base)
    const T* k_base = K + batch_idx * seq_len_kv * kv_heads * head_dim;
    const T* v_base = V + batch_idx * seq_len_kv * kv_heads * head_dim;

    // 将 Q 预加载到寄存器/本地内存
    float q_reg[MAX_HEAD_DIM]; 
    float o_reg[MAX_HEAD_DIM] = {0.0f};

    for (int d = 0; d < head_dim; ++d) {
        q_reg[d] = static_cast<float>(q_ptr[d]);
    }

    // Online Softmax
    // 遍历一次 K 和 V，同时计算 max, sum, 和 weighted output
    float max_score = -FLT_MAX;
    float denom = 0.0f; // sum of exp(score - max_score)

    for (int j = 0; j < seq_len_kv; ++j) {
        // Causal Mask 检查
        if (is_causal && j > q_pos) continue;

        const T* kj = k_base + (j * kv_heads + kv_head_idx) * head_dim;
        const T* vj = v_base + (j * kv_heads + kv_head_idx) * head_dim;

        // 计算 Q * K^T (Dot Product)
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_reg[d] * static_cast<float>(kj[d]);
        }
        score *= scale;

        if (score > max_score) {
            float diff = score - max_score;
            float rescale_factor = expf(-diff); // e^(old_max - new_max)

            // 更新 max
            max_score = score;

            // 修正旧的 denom 和 output
            denom = denom * rescale_factor + 1.0f; // +1.0 是因为当前项 e^(score - score) = 1
            
            for (int d = 0; d < head_dim; ++d) {
                o_reg[d] = o_reg[d] * rescale_factor + static_cast<float>(vj[d]);
            }
        } else {
            // 当前值比最大值小：直接计算 exp 并累加
            float diff = score - max_score;
            float val = expf(diff);
            
            denom += val;
            for (int d = 0; d < head_dim; ++d) {
                o_reg[d] += static_cast<float>(vj[d]) * val;
            }
        }
    }

    // 归一化
    T* o_ptr = O + (
        (batch_idx * seq_len_q + q_pos) * query_heads + head_idx
    ) * head_dim;

    // 防止除以0
    float inv_denom = 1.0f / (denom + 1e-6f);

    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] = static_cast<T>(o_reg[d] * inv_denom);
    }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
    
    // 安全检查：假设 head_dim <= 128，如超过请调整 MAX_HEAD_DIM
    if (head_dim > MAX_HEAD_DIM) {
        std::cerr << "Error: head_dim exceeds MAX_HEAD_DIM (128)" << std::endl;
        exit(1);
    }

    size_t o_numel = batch_size * target_seq_len * query_heads * head_dim;
    if(h_o.size() != o_numel) {
        h_o.resize(o_numel);
    }

    T *d_q, *d_k, *d_v, *d_o;
    size_t q_size = o_numel * sizeof(T);
    size_t kv_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    
    cudaMalloc(&d_q, q_size);
    cudaMalloc(&d_k, kv_size);
    cudaMalloc(&d_v, kv_size);
    cudaMalloc(&d_o, q_size);

    cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_size, cudaMemcpyHostToDevice);

    float scale = 1.0f / sqrtf((float)head_dim);
    
    // 策略：每个线程处理一个 query token
    int total_queries = batch_size * query_heads * target_seq_len;
    int threads = 256;
    int blocks = (total_queries + threads - 1) / threads;

    flash_attention_kernel_optimized<T><<<blocks, threads>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        scale, is_causal
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_o.data(), d_o, q_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);