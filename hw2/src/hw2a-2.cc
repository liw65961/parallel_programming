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
#include <emmintrin.h>

unsigned long long ncpus;
int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;
double now_row;
int *image;
int iter_row;
double space_row;
double space_col;
pthread_mutex_t lock;

void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
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
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
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

void *cal_md_set(void *arg)
{
    while (iter_row < height)
    {
        int local_iter_row;
        pthread_mutex_lock(&lock);
        local_iter_row = iter_row;
        iter_row++;
        pthread_mutex_unlock(&lock);

        double thread_row = lower + local_iter_row * space_row;
        int i1=0, i2=1;
        __m128d x0 = _mm_set_pd(i1 * space_col + left, i2 * space_col + left);
        __m128d y0 = _mm_set1_pd(thread_row);
        __m128d x = _mm_setzero_pd();
        __m128d y = _mm_setzero_pd();
        __m128d xx = _mm_mul_pd(x, x);
        __m128d yy = _mm_mul_pd(y, y);
        __m128d length_squared = _mm_setzero_pd();

        int store_repeats[2] = {0, 0};
        int it=1;
        while(i1<width||i2<width)
        { 
            __m128d temp = _mm_add_pd(_mm_sub_pd(xx, yy), x0);
            y = _mm_add_pd(_mm_mul_pd(_mm_set1_pd(2.0), _mm_mul_pd(x, y)), y0);
            x = temp;
            xx = _mm_mul_pd(x, x);
            yy = _mm_mul_pd(y, y);
            length_squared = _mm_add_pd(xx, yy);

            double len[2];
            _mm_storeu_pd(len, length_squared);

            store_repeats[0]++;
            if (store_repeats[0] >= iters || len[0] >= 4.0)
            {
                if (i1 < width) image[local_iter_row * width + i1] = store_repeats[0];
                it++;
                i1=it;
                if (i1 < width)
                {
                    x0[0]=i1 * space_col + left;
                    x[0]=0;
                    y[0]=0;
                    xx[0]=0;
                    yy[0]=0;
                    length_squared[0]=0;
                    store_repeats[0]=0;
                }
            }

            store_repeats[1]++;
            if (store_repeats[1] >= iters || len[1] >= 4.0)
            {
                if (i2 < width) image[local_iter_row * width + i2] = store_repeats[1];
                it++;
                i2=it;
                if (i2 < width)
                {
                    x0[1]=i2 * space_col + left;
                    x[1]=0;
                    y[1]=0;
                    xx[1]=0;
                    yy[1]=0;
                    length_squared[1]=0;
                    store_repeats[1]=0;
                }
            }
        }
    }
    return NULL;
}


int main(int argc, char **argv)
{

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    ncpus = CPU_COUNT(&cpu_set);
    pthread_t *threads = new pthread_t[ncpus];

    /* argument parsing */
    assert(argc == 9);
    const char *filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    int *threads_arg = new int[ncpus];

    /* allocate memory for image */
    image = (int *)malloc(width * height * sizeof(int));
    assert(image);

    pthread_mutex_init(&lock, NULL);
    space_row = (upper - lower) / height;
    space_col = (right - left) / width;
    iter_row = 0;
    for (int i = 0; i < ncpus; i++)
    {
        threads_arg[i] = i;
        pthread_create(&threads[i], nullptr, cal_md_set, (void *)&threads_arg[i]);
    }
    for (int i = 0; i < ncpus; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&lock);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
