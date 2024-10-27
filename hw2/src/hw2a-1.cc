#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <iomanip>

unsigned long long ncpus;
int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;
double now_row;
int* image;
int iter_row;
double space_row;
double space_col;
pthread_mutex_t lock;


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void *cal_md_set(void* arg)
{
    while (iter_row < height)
    {
        int local_iter_row;
        pthread_mutex_lock(&lock);
        local_iter_row = iter_row; 
        iter_row++;   
        pthread_mutex_unlock(&lock);

        double thread_row = lower + local_iter_row * space_row;
        for (int i = 0; i < width; ++i) {
            double x0 = i * space_col + left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + thread_row;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[local_iter_row * width + i] = repeats;
        }
    }
    return NULL;
}

int main(int argc, char** argv) {
    
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    ncpus = CPU_COUNT(&cpu_set);
    pthread_t *threads = new pthread_t[ncpus];

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    int *threads_arg = new int[ncpus];

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    
    pthread_mutex_init(&lock, NULL);
    space_row = (upper - lower) / height;
    space_col = (right - left) / width;
    iter_row = 0;
    for (int i = 0; i < ncpus; i++) {
        threads_arg[i] = i;
        pthread_create(&threads[i], nullptr, cal_md_set, (void*)&threads_arg[i]);
    }
    for (int i = 0; i < ncpus; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&lock);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
