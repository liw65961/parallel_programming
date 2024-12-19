#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#define BF 64


const int INF = 60001;
void input(char *inFileName);
void output(char *outFileName);
void block_FW(int B);
__global__ void phase1(int B, int r, int *device_Dist, int tn);
__global__ void phase2(int B, int r, int *device_Dist, int tn);
__global__ void phase3(int B, int r, int *device_Dist, int tn);
/* Get ceil(a / b) */
__device__ __host__ int ceil(int a, int b) {
    return (a + b - 1) / b;
}

int n, m, tn;
int *Dist, *device_Dist;
const int B = 64;

int main(int argc, char* argv[]){
    input(argv[1]);
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char *inFileName){
    FILE *file = fopen(inFileName, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    tn = ceil(n, B) * B;
    cudaMallocHost(&Dist, tn*tn*sizeof(int));
    for(int i = 0; i < tn; i++){
        for(int j = 0; j < tn; j++){
            Dist[i*tn+j] = (i==j&&i<n)?0:INF;
        }
    }

    int pair[3];
    for(int i = 0; i < m; i++){
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]*tn+pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName){
    FILE *file = fopen(outFileName, "w");
    for(int i = 0; i < n; i++){
        fwrite(&Dist[i*tn], sizeof(int), n, file);
    }
    fclose(file);
    cudaFreeHost(Dist);
}

void block_FW(int B){
    cudaMalloc(&device_Dist, tn*tn*sizeof(int));
    cudaMemcpy(device_Dist, Dist, tn*tn*sizeof(int), cudaMemcpyHostToDevice);
    int round = tn/B;
    for(int r = 0; r < round; r++){
        phase1 <<<1, dim3(32, 32)>>> (B, r, device_Dist, tn);
        phase2 <<<dim3(2, round-1), dim3(32, 32)>>> (B, r, device_Dist, tn);
        phase3 <<<dim3(round-1, round-1), dim3(32, 32)>>> (B, r, device_Dist, tn);
    }
    cudaMemcpy(Dist, device_Dist, tn*tn*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_Dist);
}

__global__ void phase1(int B, int r, int *device_Dist, int tn){
    __shared__ int s[BF*BF];
    int b_i = r*B, b_j = r*B, b_k = r*B;
    int i = threadIdx.y, j = threadIdx.x;
    int iter = ceil(B, 32);

    if(threadIdx.y>=B||threadIdx.x>=B)return;

    #pragma unroll
    for(int x = 0; x < iter; x++){
        #pragma unroll
        for(int y = 0; y < iter; y++){
            s[(i+32 * y)*B+(j+32 * x)] = device_Dist[(b_i+(i+32 * y))*tn+(b_j+(j+32 * x))];
        }
    }
    __syncthreads();

    #pragma unroll 
    for(int k = 0; k < B; k++){
        #pragma unroll
        for(int x = 0; x < iter; x++){
            #pragma unroll
            for(int y = 0; y < iter; y++){
                s[(i+32 * y)*B+(j+32 * x)] = min(s[(i+32 * y)*B+(j+32 * x)], s[(i+32 * y)*B+k]+s[k*B+(j+32 * x)]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int x = 0; x < iter; x++){
        #pragma unroll
        for(int y = 0; y < iter; y++){
            device_Dist[(b_i+(i+32 * y))*tn+(b_j+(j+32 * x))] = s[(i+32 * y)*B+(j+32 * x)];
        }
    }
}

__global__ void phase2(int B, int r, int *device_Dist, int tn){
    __shared__ int s[2*BF*BF];
    // ROW: (blockIdx.x = 1), COL: (blockIdx.y = 0)
    int b_i = (blockIdx.x*r+(!blockIdx.x)*(blockIdx.y+(blockIdx.y>=r)))*B;
    int b_j = (blockIdx.x*(blockIdx.y+(blockIdx.y>=r))+(!blockIdx.x)*r)*B;
    int b_k = r*B;
    int i = threadIdx.y, j = threadIdx.x;
    int iter = ceil(B, 32);
    int BB = B*B;
    if(threadIdx.y>=B||threadIdx.x>=B)return;

    #pragma unroll
    for(int x = 0; x < iter; x++){
        #pragma unroll
        for(int y = 0; y < iter; y++){
            s[(i+32*y)*B+(j+32*x)] = device_Dist[(b_i+(i+32*y))*tn+(b_k+(j+32*x))];
            s[BB+(i+32*y)*B+(j+32*x)] = device_Dist[(b_k+(i+32*y))*tn+(b_j+(j+32*x))];
        }
    }
    __syncthreads();

    #pragma unroll 
    for(int k = 0; k < B; k++){
        #pragma unroll
        for(int x = 0; x < iter; x++){
            #pragma unroll
            for(int y = 0; y < iter; y++){
                device_Dist[(b_i+(i+32*y))*tn+(b_j+(j+32*x))] = min(device_Dist[(b_i+(i+32*y))*tn+(b_j+(j+32*x))], s[(i+32*y)*B+k]+s[BB+k*B+(j+32*x)]);
            }
        }
    }
}

__global__ void phase3(int B, int r, int *device_Dist, int tn){
    __shared__ int s[2*BF*BF];
    int b_i = (blockIdx.x+(blockIdx.x>=r))*B, b_j = (blockIdx.y+(blockIdx.y>=r))*B, b_k = r*B;
    int i = threadIdx.y, j = threadIdx.x;
    int iter = ceil(B, 32);
    int BB = B*B;
    if(threadIdx.y>=B||threadIdx.x>=B)return;
    #pragma unroll
    for(int x = 0; x < iter; x++){
        #pragma unroll
        for(int y = 0; y < iter; y++){
            s[(i+32*y)*B+(j+32*x)] = device_Dist[(b_i+(i+32*y))*tn+(b_k+(j+32*x))];
            s[BB+(i+32*y)*B+(j+32*x)] = device_Dist[(b_k+(i+32*y))*tn+(b_j+(j+32*x))];
        }
    }
    __syncthreads();

    #pragma unroll 
    for(int k = 0; k < B; k++){
        #pragma unroll
        for(int x = 0; x < iter; x++){
            #pragma unroll
            for(int y = 0; y < iter; y++){
                device_Dist[(b_i+(i+32*y))*tn+(b_j+(j+32*x))] = min(device_Dist[(b_i+(i+32*y))*tn+(b_j+(j+32*x))], s[(i+32*y)*B+k]+s[BB+k*B+(j+32*x)]);
            }
        }
    }
}