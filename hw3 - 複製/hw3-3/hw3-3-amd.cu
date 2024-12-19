#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#include <sys/time.h>

const int INF = 60001;
void input(char *inFileName);
void output(char *outFileName);
void block_FW(int B);

__global__ void phase1(int B, int r, int *device_Dist, int tn);
__global__ void phase2(int B, int r, int *device_Dist, int tn);
__global__ void phase3(int B, int r, int *device_Dist, int tn, int start);

__device__ __host__ int ceil(int a, int b) {
    return (a + b - 1) / b;
}

int n, m, tn;
int *Dist, *device_Dist[2];
const int B = 64;

int main(int argc, char* argv[]){
    struct timeval total_start, total_end;
    gettimeofday(&total_start, NULL);

    input(argv[1]);
    block_FW(B);
    output(argv[2]);

    gettimeofday(&total_end, NULL);
    double total_time = (total_end.tv_sec - total_start.tv_sec) + 1e-6 * (total_end.tv_usec - total_start.tv_usec);
    printf("Total Execution Time: %.6f seconds.\n", total_time);

    return 0;
}

void input(char *inFileName){
    struct timeval start, end;
    gettimeofday(&start, NULL);

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

    gettimeofday(&end, NULL);
    double input_time = (end.tv_sec - start.tv_sec) + 1e-6 * (end.tv_usec - start.tv_usec);
    printf("Input Time: %.6f seconds.\n", input_time);
}

void output(char *outFileName){
    struct timeval start, end;
    gettimeofday(&start, NULL);

    FILE *file = fopen(outFileName, "w");
    for(int i = 0; i < n; i++){
        fwrite(&Dist[i*tn], sizeof(int), n, file);
    }
    fclose(file);
    cudaFreeHost(Dist);

    gettimeofday(&end, NULL);
    double output_time = (end.tv_sec - start.tv_sec) + 1e-6 * (end.tv_usec - start.tv_usec);
    printf("Output Time: %.6f seconds.\n", output_time);
}

void block_FW(int B){
    struct timeval comp_start, comp_end, mem_start, mem_end;
    double comp_time = 0, mem_copy_time = 0;

    #pragma omp parallel num_threads(2)
    {
        int id = omp_get_thread_num(), round = tn/64;
        int start = (round/2)*id, size = (round/2)+(round%2)*id;

        gettimeofday(&mem_start, NULL);
        cudaSetDevice(id);
        cudaMalloc(&device_Dist[id], tn*tn*sizeof(int));
        cudaMemcpy(device_Dist[id]+(start*64*tn), Dist+(start*64*tn), size*64*tn*sizeof(int), cudaMemcpyHostToDevice);
        gettimeofday(&mem_end, NULL);
        mem_copy_time += (mem_end.tv_sec - mem_start.tv_sec) + 1e-6 * (mem_end.tv_usec - mem_start.tv_usec);

        dim3 num_thds(32, 32);
        dim3 num_blks_ph2(2, round-1);
        dim3 num_blks_ph3(size, round-1);

        gettimeofday(&comp_start, NULL);
        for(int r = 0; r < round; r++){
            if(r<round/2)cudaMemcpyPeer(device_Dist[1]+(r*64*tn), 1, device_Dist[0]+(r*64*tn), 0, 64*tn*sizeof(int));
            else cudaMemcpyPeer(device_Dist[0]+(r*64*tn), 0, device_Dist[1]+(r*64*tn), 1, 64*tn*sizeof(int));
            #pragma omp barrier

            phase1 <<<1, num_thds>>> (B, r, device_Dist[id], tn);
            phase2 <<<num_blks_ph2, num_thds>>> (B, r, device_Dist[id], tn);
            phase3 <<<num_blks_ph3, num_thds>>> (B, r, device_Dist[id], tn, start);
        }
        gettimeofday(&comp_end, NULL);
        comp_time += (comp_end.tv_sec - comp_start.tv_sec) + 1e-6 * (comp_end.tv_usec - comp_start.tv_usec);

        gettimeofday(&mem_start, NULL);
        cudaMemcpy(Dist+(start*64*tn), device_Dist[id]+(start*64*tn), size*64*tn*sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(device_Dist[id]);
        gettimeofday(&mem_end, NULL);
        mem_copy_time += (mem_end.tv_sec - mem_start.tv_sec) + 1e-6 * (mem_end.tv_usec - mem_start.tv_usec);
    }

    printf("Computation Time: %.6f seconds.\n", comp_time);
    printf("Memory Copy Time: %.6f seconds.\n", mem_copy_time);
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

__global__ void phase3(int B, int r, int *device_Dist, int tn, int start){
    __shared__ int s[2*64*64];
    int blk_i = (start+blockIdx.x)<<6, blk_j = (blockIdx.y+(blockIdx.y>=r))<<6, blk_p = r<<6;
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