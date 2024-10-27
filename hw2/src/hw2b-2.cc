#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <png.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <emmintrin.h>

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


int main(int argc, char** argv) {

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    int* total_image = (int*)malloc(width * height * sizeof(int));
    memset(image, 0, width * height * sizeof(int));
    memset(total_image, 0, width * height * sizeof(int));
    assert(image);

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    double space_row=(upper - lower) / height;
    double space_col=(right - left) / width;

	/* mandelbrot set */
    #pragma omp parallel for schedule(dynamic)
    for (int j = rank; j < height; j+=size) {
        int i1=0, i2=1;
        __m128d x0 = _mm_set_pd(i1 * space_col + left, i2 * space_col + left);
        __m128d y0 = _mm_set1_pd(j * space_row + lower);
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
                if (i1 < width) image[j * width + i1] = store_repeats[0];
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
                if (i2 < width) image[j * width + i2] = store_repeats[1];
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

	MPI_Reduce(image, total_image, height*width, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	if (rank == 0) 
    {
		/* draw and cleanup */
        write_png(filename, iters, width, height, total_image);
        free(total_image);
        free(image);
	}
}
