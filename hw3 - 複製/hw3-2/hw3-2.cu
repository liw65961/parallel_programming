#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

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

    tn = ceil(n, 64) * 64;
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
    int round = tn/64;
    dim3 num_thds(32, 32);
    dim3 num_blks_ph2(2, round-1);
    dim3 num_blks_ph3(round-1, round-1);
    for(int r = 0; r < round; r++){
        phase1 <<<1, num_thds>>> (B, r, device_Dist, tn);
        phase2 <<<num_blks_ph2, num_thds>>> (B, r, device_Dist, tn);
        phase3 <<<num_blks_ph3, num_thds>>> (B, r, device_Dist, tn);
    }
    cudaMemcpy(Dist, device_Dist, tn*tn*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_Dist);
}

__global__ void phase1(int B, int r, int *device_Dist, int tn){
    __shared__ int s[64*64];
    int blk_i = r<<6, blk_j = r<<6;
    int i = threadIdx.y, j = threadIdx.x;

    #pragma unroll
    for(int x = 0; x < 2; x++){
        #pragma unroll
        for(int y = 0; y < 2; y++){
            s[(i+32 * y)*64+(j+32 * x)] = device_Dist[(blk_i+(i+32 * y))*tn+(blk_j+(j+32 * x))];
        }
    }
    __syncthreads();

    #pragma unroll 48
    for(int k = 0; k < 64; k++){
        #pragma unroll
        for(int x = 0; x < 2; x++){
            #pragma unroll
            for(int y = 0; y < 2; y++){
                s[(i+32 * y)*64+(j+32 * x)] = min(s[(i+32 * y)*64+(j+32 * x)], s[(i+32 * y)*64+k]+s[k*64+(j+32 * x)]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int x = 0; x < 2; x++){
        #pragma unroll
        for(int y = 0; y < 2; y++){
            device_Dist[(blk_i+(i+32 * y))*tn+(blk_j+(j+32 * x))] = s[(i+32 * y)*64+(j+32 * x)];
        }
    }
}

__global__ void phase2(int B, int r, int *device_Dist, int tn){
    __shared__ int s[2*64*64];
    int blk_i = (blockIdx.x*r+(!blockIdx.x)*(blockIdx.y+(blockIdx.y>=r)))<<6;
    int blk_j = (blockIdx.x*(blockIdx.y+(blockIdx.y>=r))+(!blockIdx.x)*r)<<6;
    int blk_p = r<<6;
    int i = threadIdx.y, j = threadIdx.x;

    int val0 = device_Dist[(blk_i+i)*tn+(blk_j+j)];
    int val1 = device_Dist[(blk_i+i)*tn+(blk_j+(j+32))];
    int val2 = device_Dist[(blk_i+(i+32))*tn+(blk_j+j)];
    int val3 = device_Dist[(blk_i+(i+32))*tn+(blk_j+(j+32))];


    #pragma unroll
    for(int x = 0; x < 2; x++){
        #pragma unroll
        for(int y = 0; y < 2; y++){
            s[(i+32*y)*64+(j+32*x)] = device_Dist[(blk_i+(i+32*y))*tn+(blk_p+(j+32*x))];
            s[4096+(i+32*y)*64+(j+32*x)] = device_Dist[(blk_p+(i+32*y))*tn+(blk_j+(j+32*x))];
        }
    }
    __syncthreads();
    #pragma unroll 48
    for(int k = 0; k < 64; k++){
        val0 = min(val0, s[i*64+k]+s[4096+k*64+j]);
        val1 = min(val1, s[i*64+k]+s[4096+k*64+(j+32)]);
        val2 = min(val2, s[(i+32)*64+k]+s[4096+k*64+j]);
        val3 = min(val3, s[(i+32)*64+k]+s[4096+k*64+(j+32)]);
    }

    device_Dist[(blk_i+i)*tn+(blk_j+j)] = val0;
    device_Dist[(blk_i+i)*tn+(blk_j+(j+32))] = val1;
    device_Dist[(blk_i+(i+32))*tn+(blk_j+j)] = val2;
    device_Dist[(blk_i+(i+32))*tn+(blk_j+(j+32))] = val3;
}

__global__ void phase3(int B, int r, int *device_Dist, int tn){
    __shared__ int s[2*64*64];
    int blk_i = (blockIdx.x+(blockIdx.x>=r))<<6, blk_j = (blockIdx.y+(blockIdx.y>=r))<<6, blk_p = r<<6;
    int i = threadIdx.y, j = threadIdx.x;

    int val0 = device_Dist[(blk_i+i)*tn+(blk_j+j)];
    int val1 = device_Dist[(blk_i+i)*tn+(blk_j+(j+32))];
    int val2 = device_Dist[(blk_i+(i+32))*tn+(blk_j+j)];
    int val3 = device_Dist[(blk_i+(i+32))*tn+(blk_j+(j+32))];

    #pragma unroll
    for(int x = 0; x < 2; x++){
        #pragma unroll
        for(int y = 0; y < 2; y++){
            s[(i+32*y)*64+(j+32*x)] = device_Dist[(blk_i+(i+32*y))*tn+(blk_p+(j+32*x))];
            s[4096+(i+32*y)*64+(j+32*x)] = device_Dist[(blk_p+(i+32*y))*tn+(blk_j+(j+32*x))];
        }
    }

    __syncthreads();
    #pragma unroll 48
    for(int k = 0; k < 64; k++){
        val0 = min(val0, s[i*64+k]+s[4096+k*64+j]);
        val1 = min(val1, s[i*64+k]+s[4096+k*64+(j+32)]);
        val2 = min(val2, s[(i+32)*64+k]+s[4096+k*64+j]);
        val3 = min(val3, s[(i+32)*64+k]+s[4096+k*64+(j+32)]);
    }

    device_Dist[(blk_i+i)*tn+(blk_j+j)] = val0;
    device_Dist[(blk_i+i)*tn+(blk_j+(j+32))] = val1;
    device_Dist[(blk_i+(i+32))*tn+(blk_j+j)] = val2;
    device_Dist[(blk_i+(i+32))*tn+(blk_j+(j+32))] = val3;
}