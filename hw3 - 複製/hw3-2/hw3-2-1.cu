#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const int INF = ((1 << 30) - 1);

int n, m, matrix_size;
int* Dist;
const int threads_number=32;

__constant__ int device_matrix_size, device_threads_number;

__device__ __host__ int twoD_to_oneD_index(int, int, int);
__device__ __host__ int ceil(int, int);
void input(char*);
void output(char*);
__global__ void ph1_iter(int*, int);
__global__ void ph2_iter(int*, int);
__global__ void ph3_iter(int*, int);
void block_FW(int*);


int main(int argc, char* argv[]) {
    // Read input
    input(argv[1]);

    // Allocate memory for constants
    cudaMemcpyToSymbol(device_matrix_size, &matrix_size, sizeof(int));
    cudaMemcpyToSymbol(device_threads_number, &threads_number, sizeof(int));

    // Allocate memory for device_Dist
    int* device_Dist;
    cudaMalloc((void**)&device_Dist, sizeof(int) * matrix_size * matrix_size);
    cudaMemcpy(device_Dist, Dist, sizeof(int) * matrix_size * matrix_size, cudaMemcpyHostToDevice);

    // Block FW
    block_FW(device_Dist);

    // Copy data from device to host
    cudaMemcpy(Dist, device_Dist, sizeof(int) * matrix_size * matrix_size, cudaMemcpyDeviceToHost);

    // Write output
    output(argv[2]);
    return 0;
}

/* Read file input */
void input(char* infile) {
    // Read vertex num and edge num
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    // Calculate matrix size
    matrix_size = ceil(n, threads_number) * threads_number;

    // Allocate memory for Dist
    cudaMallocHost((void**)&Dist, sizeof(int) * matrix_size * matrix_size);

    // Initialize Dist
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            int idx = twoD_to_oneD_index(i, j, matrix_size);
            if(i == j && i < n && j < n)
                Dist[idx] = 0;
            else
                Dist[idx] = INF;
        }
    }

    // Read edges
    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        int idx = twoD_to_oneD_index(pair[0], pair[1], matrix_size);
        Dist[idx] = pair[2];
    }
    fclose(file);
}

/* BLocked Floyd-Warshall */
void block_FW(int* device_Dist) {
    int round = ceil(n, threads_number);
    int s_mem_size = threads_number * threads_number * sizeof(int);
    dim3 thds_per_blk(threads_number, threads_number);
    dim3 p2_blks_per_grid(2, round - 1); 
    dim3 p3_blks_per_grid(round - 1, round - 1);

    for (int r = 0; r < round; ++r) {
        ph1_iter<<<1, thds_per_blk, s_mem_size>>>(device_Dist, r);
        ph2_iter<<<p2_blks_per_grid, thds_per_blk, 2 * s_mem_size>>>(device_Dist, r);
        ph3_iter<<<p3_blks_per_grid, thds_per_blk, 3 * s_mem_size>>>(device_Dist, r);
    }
}

/* Phase 1's kernel */
__global__ void ph1_iter(int* device_Dist, int r) {
    int coord_i=threadIdx.y+r*device_threads_number;
    int coord_j=threadIdx.x+r*device_threads_number;

    int shmem_idx = coord_i*device_matrix_size+coord_j;
    // extern __shared__ int sh_mem[];
    // sh_mem[shmem_idx]=device_Dist[shmem_idx];

    for(int k = r*device_threads_number; k < (r+1)*device_threads_number; ++k)
    {
        int dist_ik=device_Dist[coord_i*device_matrix_size+k];
        int dist_kj=device_Dist[k*device_matrix_size+coord_j];
        if(device_Dist[shmem_idx]>dist_ik+dist_kj)device_Dist[shmem_idx]=dist_ik+dist_kj;
        __syncthreads();
    }
    // device_Dist[shmem_idx]=sh_mem[shmem_idx];

}

/* Phase 2's kernel */
__global__ void ph2_iter(int* device_Dist, int r) {
    int coord_i;
    int coord_j;
    int shmem_idx;
    // extern __shared__ int sh_mem[];
    // sh_mem[shmem_idx]=device_Dist[shmem_idx];
    if(blockIdx.y < r)
    {
        if(blockIdx.x == 0)
        {
            coord_i=threadIdx.y+r*device_threads_number;
            coord_j=threadIdx.x+blockIdx.y*device_threads_number;
        }
        else if(blockIdx.x == 1)
        {
            coord_i=threadIdx.y+blockIdx.y*device_threads_number;
            coord_j=threadIdx.x+r*device_threads_number;
        }
    }
    else
    {
        if(blockIdx.x == 0)
        {
            coord_i=threadIdx.y+r*device_threads_number;
            coord_j=threadIdx.x+(blockIdx.y+1)*device_threads_number;
        }
        else if(blockIdx.x == 1)
        {
            coord_i=threadIdx.y+(blockIdx.y+1)*device_threads_number;
            coord_j=threadIdx.x+r*device_threads_number;
        }
    }
    shmem_idx=coord_i*device_matrix_size+coord_j;
    for(int k = r*device_threads_number; k < (r+1)*device_threads_number; ++k)
    {
        int dist_ik=device_Dist[coord_i*device_matrix_size+k];
        int dist_kj=device_Dist[k*device_matrix_size+coord_j];
        if(device_Dist[shmem_idx]>dist_ik+dist_kj)device_Dist[shmem_idx]=dist_ik+dist_kj;
        __syncthreads();
    }
    // device_Dist[shmem_idx]=sh_mem[shmem_idx];
}

/* Phase 3's kernel */
__global__ void ph3_iter(int* device_Dist, int r) {
    int coord_i;
    int coord_j;

    int shmem_idx;
    // extern __shared__ int sh_mem[];
    // sh_mem[shmem_idx]=device_Dist[shmem_idx];
    if(blockIdx.x < r&&blockIdx.y < r) 
    {
        coord_i=blockIdx.y*device_threads_number+threadIdx.y;
        coord_j=blockIdx.x*device_threads_number+threadIdx.x;
    }
    else if(blockIdx.x >= r&&blockIdx.y < r) 
    {
        coord_i=blockIdx.y*device_threads_number+threadIdx.y;
        coord_j=(blockIdx.x+1)*device_threads_number+threadIdx.x;
    }
    else if(blockIdx.x < r && blockIdx.y >= r) 
    {
        coord_i=(blockIdx.y+1)*device_threads_number+threadIdx.y;
        coord_j=blockIdx.x*device_threads_number+threadIdx.x;
    }
    else 
    {
        coord_i=(blockIdx.y+1)*device_threads_number+threadIdx.y;
        coord_j=(blockIdx.x+1)*device_threads_number+threadIdx.x;
    }
    shmem_idx=coord_i*device_matrix_size+coord_j;
    for(int k = r*device_threads_number; k < (r+1)*device_threads_number; ++k)
    {
        int dist_ik=device_Dist[coord_i*device_matrix_size+k];
        int dist_kj=device_Dist[k*device_matrix_size+coord_j];
        if(device_Dist[shmem_idx]>dist_ik+dist_kj)device_Dist[shmem_idx]=dist_ik+dist_kj;
        __syncthreads();
    }
    // device_Dist[shmem_idx]=sh_mem[shmem_idx];

}

/* Write file output */
void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int idx = twoD_to_oneD_index(i, j, matrix_size);
            if(Dist[idx] >= INF)
                Dist[idx] = INF;
        }
        fwrite(Dist + i * matrix_size, sizeof(int), n, outfile);
    }
    fclose(outfile);
}

__device__ __host__ int twoD_to_oneD_index(int i, int j, int row_size) {
    return i * row_size + j;
}

/* Get ceil(a / b) */
__device__ __host__ int ceil(int a, int b) {
    return (a + b - 1) / b;
}
