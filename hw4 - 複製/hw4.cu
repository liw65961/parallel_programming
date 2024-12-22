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
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d; // B : batch size, N rows, d columns
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();

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
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

__global__ void QKDotAndScalar(float *q, float *k, float *out, int d, float scalar, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0F;
        for (int t = 0; t < d; t++) {
            value += q[row * d + t] * k[col * d + t];
        }
        out[row * N + col] = value * scalar;
    }
}

__global__ void RowMax(float *in, float *out, int N, int br) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < br; j++) {
            max_val = _max(max_val, in[row * br + j]);
        }
        out[row] = max_val;
    }
}

__global__ void MinusMaxAndExp(float *in, float *out, float *max_vals, int N, int br) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < br) {
        out[row * br + col] = exp(in[row * br + col] - max_vals[row]);
    }
}

__global__ void RowSum(float *in, float *out, int N, int br) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0F;
        for (int j = 0; j < br; j++) {
            sum += in[row * br + j];
        }
        out[row] = sum;
    }
}

__global__ void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < br && col < d) {
        float mi_new = _max(mi[row], mij[row]);
        float li_new = exp(mi[row] - mi_new) * li[row] + exp(mij[row] - mi_new) * lij[row];

        float pv = 0.0F;
        for (int t = 0; t < br; t++) {
            pv += pij[row * br + t] * vj[t * d + col];
        }
        oi[row * d + col] = (li[row] * exp(mi[row] - mi_new) * oi[row * d + col] + exp(mij[row] - mi_new) * pv) / li_new;

        if (col == 0) {  // Only update `mi` and `li` once per row
            mi[row] = mi_new;
            li[row] = li_new;
        }
    }
}

void flash_attention(float *q, float *k, float *v, float *o, int N, int d) {
    float *d_q, *d_k, *d_v, *d_o;
    float *d_sij, *d_pij, *d_mij, *d_lij, *d_m, *d_l;

    int size_q = N * d * sizeof(float);
    int size_o = N * d * sizeof(float);
    int size_temp = N * N * sizeof(float);

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

    cudaMemset(d_m, 0.0F, N * sizeof(float));
    cudaMemset(d_l, 0.0F, N * sizeof(float));


    dim3 blockSize(32, 32);
    dim3 gridSize((N + 31) / 32, (N + 31) / 32);

    QKDotAndScalar<<<gridSize, blockSize>>>(d_q, d_k, d_sij, d, 1.0 / sqrt(d), N);
    RowMax<<<(N + 1023) / 1024, 1024>>>(d_sij, d_mij, N, N);
    MinusMaxAndExp<<<gridSize, blockSize>>>(d_sij, d_pij, d_mij, N, N);
    RowSum<<<(N + 1023) / 1024, 1024>>>(d_pij, d_lij, N, N);
    UpdateMiLiOi<<<gridSize, blockSize>>>(d_m, d_l, d_o, d_mij, d_lij, d_pij, d_v, N, d);

    cudaMemcpy(o, d_o, size_o, cudaMemcpyDeviceToHost);

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