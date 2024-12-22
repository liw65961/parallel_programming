#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o, int N, int d);

__global__ void QKDotAndScalar(float *out, float *q, float *kj, int br, int bc, float scalar);
__global__ void RowMax(float *out, float *in, int br, int bc);
__global__ void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc);
__global__ void RowSum(float *out, float *in, int br, int bc);
__global__ void UpdateMiLiOi(float *m, float *l, float *o, float *mij, float *lij, float *pij, float *vj, int br, int bc);

__device__ __host__ float _max(float a, float b) { return a > b ? a : b; }
__device__ __host__ float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_usec / 1000000 + tv.tv_sec;
}

int B, N, d; // B : batch size, N rows, d columns
float *Q, *K, *V, *O;

double start_io, end_io, total_io_time = 0;
double start_mem, end_mem, total_mem_time = 0;
double start_compute, end_compute, total_compute_time = 0;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    // Input I/O timing
    start_io = getTimeStamp();
    input(argv[1]);
    end_io = getTimeStamp();
    total_io_time += (end_io - start_io);

    double start, end;
    start = getTimeStamp();

    // Compute timing
    // start_compute = getTimeStamp();
    for (int i = 0; i < B; i++) {
        flash_attention(
            Q + (i * N * d), 
            K + (i * N * d), 
            V + (i * N * d), 
            O + (i * N * d),
            N,
            d
        );
    }

    end = getTimeStamp();

    // Output I/O timing
    start_io = getTimeStamp();
    output(argv[2]);
    end_io = getTimeStamp();
    total_io_time += (end_io - start_io);

    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Total time: %.3f seconds\n", end - start);
    printf("I/O time: %.3f seconds\n", total_io_time);
    printf("Memory operation time: %.3f seconds\n", total_mem_time);
    printf("Computation time: %.3f seconds\n", total_compute_time);

    return 0;
}

void input(char *input_filename) {
    double start_mem, end_mem;
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    start_mem = getTimeStamp();
    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));
    end_mem = getTimeStamp();
    total_mem_time += (end_mem - start_mem);

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    double start_mem, end_mem;
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    start_mem = getTimeStamp();
    free(Q);
    free(K);
    free(V);
    free(O);
    end_mem = getTimeStamp();
    total_mem_time += (end_mem - start_mem);

    fclose(file);
}


__global__ void QKDotAndScalar(float *q, float *k, float *out, int d, float scalar, int N) {
    int i = threadIdx.y;
    int j = threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    extern __shared__ float shared_mem[];
    float* shared_q = shared_mem;
    float* shared_k = shared_q + 32 * d;
    
    #pragma unroll 32
    for (int t = 0; t < d; t++) {
        shared_q[i * d + t] = q[row * d + t];
        shared_k[j * d + t] = k[col * d + t];
    }
    __syncthreads();
    float value = 0.0F;

    for (int t = 0; t < d; t++) {
        value += shared_q[i * d + t] * shared_k[j * d + t];
    }
    out[row * N + col] = value * scalar;
}

__global__ void RowMax(float *in, float *out, int N, int br) {
    int i = threadIdx.x;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    extern __shared__ float shared_mem[];
    float *shared_in = shared_mem;
    float max_val = -FLT_MAX;
    #pragma unroll 128
    for (int j = 0; j < N; j++){
        shared_in[i * N + j] = in[row * N + j];
    }
    __syncthreads();
    #pragma unroll 128
    for (int j = 0; j < N; j++) {
        max_val = _max(max_val, shared_in[i * N + j]);
    }
    out[row] = max_val;
}


__global__ void MinusMaxAndExp(float *in, float *out, float *max_vals, int N, int br) {
    int i = threadIdx.y;
    int j = threadIdx.x;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N) return;
    extern __shared__ float shared_mem[];
    float *shared_in = shared_mem;
    shared_in[i*32+j] = in[row * N + col];
    __syncthreads();
    out[row * N + col] = exp(shared_in[i*32+j] - max_vals[row]);
}

__global__ void RowSum(float *in, float *out, int N, int br) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0F;
        #pragma unroll 128
        for (int j = 0; j < br; j++) {
            sum += in[row * br + j];
        }
        out[row] = sum;
    }
}

__global__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int N, int d) {
    int i = threadIdx.y;
    int j = threadIdx.x;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= d) return;
    extern __shared__ float shared_mem[];
    float *shared_mi = shared_mem;
    float *shared_mij = shared_mi + 32;
    shared_mi[i] = mi[row];
    shared_mij[i] = mij[row];
    float *shared_li = shared_mij + 32;
    float *shared_lij = shared_li + 32;
    shared_li[i] = li[row];
    shared_lij[i] = lij[row];
    float *shared_pij = shared_lij + 32;
    #pragma unroll 128
    for (int t = 0; t < N; t++){
        shared_pij[i * N + t] = pij[row * N + t];
    }
    __syncthreads();
    float mi_new = _max(shared_mi[i], shared_mij[i]);
    float li_new = exp(shared_mi[i] - mi_new) * shared_li[i] + exp(shared_mij[i] - mi_new) * shared_lij[i];
    
    float pv = 0.0F;
    #pragma unroll 128
    for (int t = 0; t < N; t++) {
        pv += shared_pij[i * N + t] * vj[t * d + col];
    }
    oi[row * d + col] = (shared_li[i] * exp(shared_mi[i] - mi_new) * oi[row * d + col] + exp(shared_mij[i] - mi_new) * pv) / li_new;

    if (col == 0) {  // Only update `mi` and `li` once per row
        mi[row] = mi_new;
        li[row] = li_new;
    }
}

void flash_attention(float *q, float *k, float *v, float *o, int N, int d) {
    float *d_q, *d_k, *d_v, *d_o;
    float *d_sij, *d_pij, *d_mij, *d_lij, *d_m, *d_l;

    int size_q = N * d * sizeof(float);
    int size_o = N * d * sizeof(float);
    int size_temp = N * N * sizeof(float);

    start_mem = getTimeStamp();
    cudaMalloc(&d_q, size_q);
    cudaMalloc(&d_k, size_q);
    cudaMalloc(&d_v, size_q);
    cudaMalloc(&d_o, size_o);
    cudaMalloc(&d_sij, size_temp);
    cudaMalloc(&d_pij, size_temp);
    cudaMalloc(&d_mij, N * sizeof(float));
    cudaMalloc(&d_lij, N * sizeof(float));
    cudaMalloc(&d_m, N * sizeof(float));
    cudaMalloc(&d_l, N * sizeof(float));
    

    cudaMemcpy(d_q, q, size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_o, o, size_o, cudaMemcpyHostToDevice);
    end_mem = getTimeStamp();
    total_mem_time += (end_mem - start_mem);

    cudaMemset(d_m, 0.0F, N * sizeof(float));
    cudaMemset(d_l, 0.0F, N * sizeof(float));


    dim3 blockSize(32, 32);
    dim3 gridSize((N + 31) / 32, (N + 31) / 32);
    start_compute = getTimeStamp();
    int sharedMemSize_QKD = 2 * 32 * d * sizeof(float);
    QKDotAndScalar<<<gridSize, blockSize, sharedMemSize_QKD>>>(d_q, d_k, d_sij, d, 1.0 / sqrt(d), N);
    // cudaDeviceSynchronize();
    int sharedMemSize_Row = 1024 * N * sizeof(float);
    RowMax<<<(N + 1023) / 1024, 1024, sharedMemSize_Row>>>(d_sij, d_mij, N, N);
    // RowMax<<<(N + 1023) / 1024, 1024>>>(d_sij, d_mij, N, N);
    // cudaDeviceSynchronize();
    int sharedMemSize_Min = 1024 * sizeof(float);
    MinusMaxAndExp<<<gridSize, blockSize, sharedMemSize_Min>>>(d_sij, d_pij, d_mij, N, N);
    // cudaDeviceSynchronize();
    // int sharedMemSize_Row = 1024 * N * sizeof(float);
    RowSum<<<(N + 1023) / 1024, 1024>>>(d_pij, d_lij, N, N);
    // cudaDeviceSynchronize();
    int sharedMemSize_Upd = (4 * 32 + N *32) * sizeof(float);
    UpdateMiLiOi<<<gridSize, blockSize, sharedMemSize_Upd>>>(d_m, d_l, d_o, d_mij, d_lij, d_pij, d_v, N, d);
    // cudaDeviceSynchronize();
    end_compute = getTimeStamp();
    total_compute_time += (end_compute - start_compute);

    start_mem = getTimeStamp();
    cudaMemcpy(o, d_o, size_o, cudaMemcpyDeviceToHost);
    end_mem = getTimeStamp();
    total_mem_time += (end_mem - start_mem);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_sij);
    cudaFree(d_pij);
    cudaFree(d_mij);
    cudaFree(d_lij);
    cudaFree(d_m);
    cudaFree(d_l);
}