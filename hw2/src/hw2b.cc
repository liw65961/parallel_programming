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
        int vn=8;
        int it=vn-1;
        int break_while=1;
        int iv[vn];
        for(int i=0;i<vn;i++)iv[i]=i;
        double ax0[vn];
        memset(ax0, 0, vn * sizeof(double));
        for(int i=0;i<vn;i++)ax0[i]=iv[i] * space_col + left;
        __m512d x0 = _mm512_loadu_pd(ax0);
        __m512d y0 = _mm512_set1_pd(j * space_row + lower);
        __m512d x = _mm512_setzero_pd();
        __m512d y = _mm512_setzero_pd();
        __m512d xx = _mm512_mul_pd(x, x);
        __m512d yy = _mm512_mul_pd(y, y);
        __m512d length_squared = _mm512_setzero_pd();
        int store_repeats[vn];
        memset(store_repeats, 0, vn * sizeof(int));
        
        while(break_while)
        { 
            __m512d temp = _mm512_add_pd(_mm512_sub_pd(xx, yy), x0);
            y = _mm512_add_pd(_mm512_mul_pd(_mm512_set1_pd(2.0), _mm512_mul_pd(x, y)), y0);
            x = temp;
            xx = _mm512_mul_pd(x, x);
            yy = _mm512_mul_pd(y, y);
            length_squared = _mm512_add_pd(xx, yy);

            double len[vn];
            _mm512_storeu_pd(len, length_squared);
            for(int i=0;i<vn;i++)
            {
                store_repeats[i]++;
                if (store_repeats[i] >= iters || len[i] >= 4.0)
                {
                    if (iv[i] < width) image[j * width + iv[i]] = store_repeats[i];
                    it++;
                    iv[i]=it;
                    if (iv[i] < width)
                    {
                        x0[i]=iv[i] * space_col + left;
                        x[i]=0;
                        y[i]=0;
                        xx[i]=0;
                        yy[i]=0;
                        length_squared[i]=0;
                        store_repeats[i]=0;
                    }
                }
            }
            
            for(int i=0;i<vn;i++)
            {
                if(iv[i]<width)break;
                else if(i==vn-1)break_while=0;
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
