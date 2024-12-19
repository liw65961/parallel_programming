#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>

const int INF = 60001;
void input(char *inFileName);
void output(char *outFileName);
void block_FW(int B);
__global__ void phase1(int B, int r, int *device_Dist, int tn);
__global__ void phase2(int B, int r, int *device_Dist, int tn);
__global__ void phase3(int B, int r, int *device_Dist, int tn);

int n, m, tn;
int *Dist, *device_Dist;
const int B = 64;
struct timeval comp_start, comp_end, mem_start, mem_end;

int main(int argc, char* argv[]){
    struct timeval io_start, io_end;
    struct timeval total_start, total_end;
    gettimeofday(&total_start, NULL);
    gettimeofday(&io_start, NULL);
    input(argv[1]);
    gettimeofday(&io_end, NULL);

    block_FW(B);

    gettimeofday(&io_start, NULL);
    output(argv[2]);
    gettimeofday(&io_end, NULL);
    gettimeofday(&total_end, NULL);

    printf("I/O Time: %.6f seconds\n", (io_end.tv_sec + io_end.tv_usec / 1e6) - (io_start.tv_sec + io_start.tv_usec / 1e6));
    double totalTime = (total_end.tv_sec - total_start.tv_sec) * 1000.0 + (total_end.tv_usec - total_start.tv_usec) / 1000.0; // Convert to milliseconds
    printf("Total Time: %f ms\n", totalTime);
    return 0;
}

void input(char *inFileName) {
    FILE *file = fopen(inFileName, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    tn = (n + B - 1) / B * B;
    cudaMallocHost(&Dist, tn * tn * sizeof(int));
    for(int i = 0; i < tn; i++){
        for(int j = 0; j < tn; j++){
            Dist[i * tn + j] = (i == j && i < n) ? 0 : INF;
        }
    }

    int pair[3];
    for(int i = 0; i < m; i++){
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * tn + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char *outFileName) {
    FILE *file = fopen(outFileName, "w");
    for(int i = 0; i < n; i++){
        fwrite(&Dist[i * tn], sizeof(int), n, file);
    }
    fclose(file);
    cudaFreeHost(Dist);
}

void block_FW(int B){
    gettimeofday(&mem_start, NULL);
    cudaMalloc(&device_Dist, tn * tn * sizeof(int));
    cudaMemcpy(device_Dist, Dist, tn * tn * sizeof(int), cudaMemcpyHostToDevice);
    gettimeofday(&mem_end, NULL);

    int round = tn / B;
    gettimeofday(&comp_start, NULL);
    for(int r = 0; r < round; r++){
        phase1 <<<1, dim3(B, B)>>>(B, r, device_Dist, tn);
        phase2 <<<dim3(2, round-1), dim3(B, B)>>>(B, r, device_Dist, tn);
        phase3 <<<dim3(round-1, round-1), dim3(B, B)>>>(B, r, device_Dist, tn);
    }
    gettimeofday(&comp_end, NULL);

    gettimeofday(&mem_start, NULL);
    cudaMemcpy(Dist, device_Dist, tn * tn * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_Dist);
    gettimeofday(&mem_end, NULL);

    double mem_time = (mem_end.tv_sec + mem_end.tv_usec / 1e6) - (mem_start.tv_sec + mem_start.tv_usec / 1e6);
    double comp_time = (comp_end.tv_sec + comp_end.tv_usec / 1e6) - (comp_start.tv_sec + comp_start.tv_usec / 1e6);
    printf("Computation Time: %.6f seconds\n", comp_time);
    printf("Memory Copy Time: %.6f seconds\n", mem_time);
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