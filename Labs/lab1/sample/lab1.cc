#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long sqr_r = r*r;

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	for (unsigned long long x = rank; x < r; x += size) {
		unsigned long long y = ceil(sqrtl(sqr_r - x*x));
		pixels += y;
		pixels %= k;
	}
	// int sz=(int)r/size;
	// unsigned long long st = (unsigned long long)rank*(unsigned long long)size;
	// unsigned long long de = ((unsigned long long)rank+1)*(unsigned long long)size;
	// if(de>r) de=r;
	// int it=0;
	

	// for (unsigned long long x = st; x < de; x++) {
	// 	pixels = ceil(sqrtl(r*r - x*x));
	// 	pixels %= k;
	// }

	unsigned long long sum = 0;
	MPI_Reduce(&pixels, &sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	if (rank == 0) {
		unsigned long long ans = (4 * sum) % k;
		printf("%llu\n", ans);
	}
	return 0;
}

