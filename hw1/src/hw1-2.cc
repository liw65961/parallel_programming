#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <mpi.h>

int main(int argc, char **argv)
{
    using namespace std;
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_File input_file, output_file;

    int remainder = n % size;
    int data_size = n / size;
    if (rank < remainder)
        data_size++;
    int ln_data_size = data_size;
    if (rank == remainder)
        ln_data_size++;
    int rn_data_size = data_size;
    if (rank == remainder - 1)
        rn_data_size--;

    int data_location = n / size * rank;
    if (rank < remainder)
        data_location += rank;
    else
        data_location += remainder;

    float *buffer = new float[data_size], *rn_buffer = new float[rn_data_size], *ln_buffer = new float[ln_data_size], *tp_buffer = new float[data_size];

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * data_location, buffer, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    sort(buffer, buffer + data_size);

    int position = 0; // left 0, right 1;
    if (rank % 2)
        position = 1;
    int it = size+1;

    while (it--)
    {
        if (position == 0 && data_size > 0 && rank != size - 1 && rn_data_size > 0) // give to right
        {
            MPI_Send(buffer+data_size-1, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(rn_buffer, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(buffer[data_size-1]>rn_buffer[0])
            {
                MPI_Send(buffer, data_size-1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(rn_buffer+1, rn_data_size-1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int id = 0, rb_id = 0, tp_id = 0;
                while (id < data_size && rb_id < rn_data_size && tp_id < data_size)
                {
                    if (buffer[id] < rn_buffer[rb_id])
                    {
                        tp_buffer[tp_id] = buffer[id];
                        id++;
                        tp_id++;
                    }
                    else
                    {
                        tp_buffer[tp_id] = rn_buffer[rb_id];
                        rb_id++;
                        tp_id++;
                    }
                }
                if (tp_id < data_size && rb_id == rn_data_size)
                {
                    while (tp_id < data_size)
                    {
                        tp_buffer[tp_id] = buffer[id];
                        tp_id++;
                        id++;
                    }
                }
                swap(tp_buffer, buffer);
            }
        }
        else if (position == 1 && data_size > 0 && rank != 0 && ln_data_size > 0)
        {
            MPI_Send(buffer, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(ln_buffer+ln_data_size-1, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(buffer[0]<ln_buffer[ln_data_size-1])
            {
                MPI_Send(buffer+1, data_size-1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(ln_buffer, ln_data_size-1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int id = data_size - 1, lb_id = ln_data_size - 1, tp_id = data_size - 1;
                while (id > -1 && lb_id > -1 && tp_id > -1)
                {
                    if (buffer[id] > ln_buffer[lb_id])
                    {
                        tp_buffer[tp_id] = buffer[id];
                        id--;
                        tp_id--;
                    }
                    else
                    {
                        tp_buffer[tp_id] = ln_buffer[lb_id];
                        lb_id--;
                        tp_id--;
                    }
                }
                if (tp_id > -1 && lb_id == -1)
                {
                    while (tp_id > -1)
                    {
                        tp_buffer[tp_id] = buffer[id];
                        tp_id--;
                        id--;
                    }
                }
                swap(tp_buffer, buffer);
            }
        }
        if (position == 0)
            position = 1;
        else
            position = 0;
    }

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * data_location, buffer, data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}