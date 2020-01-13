#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <vector>

#define MAX_Z 1000000
#define MIN_Z 1

template< class RandomIt >
void mergesort(RandomIt first, RandomIt last) {
    size_t diff = last - first;
    if(diff < 32) {
        std::sort(first, last);
        return;
    }
    mergesort(first, first + diff/2);
    mergesort(first + diff/2, last);
    
    std::inplace_merge(first, first + diff/2, last);
}

template< class RandomIt >
void parallel_mergesort(RandomIt first, RandomIt last, int group, int rank, int max_rank, MPI_Comm comm_world) {
    int child_rank = rank + (1<<group);
    if (child_rank > max_rank) {
        mergesort(first, last);
    }
    else {
        size_t diff = last - first;
        
        MPI_Request request; MPI_Status status;
        MPI_Isend(first, diff/2, MPI_INT, child_rank, 666, comm_world, &request);
        
        parallel_mergesort(first + diff/2, last, group+1, rank, max_rank, comm_world);
        
        MPI_Recv(first, diff/2, MPI_INT, child_rank, 666, comm_world, &status);
        
        std::inplace_merge(first, first + diff/2, last);
    }
    return;
}

template< class RandomIt >
void starter_root (RandomIt first, RandomIt last, int max_rank, MPI_Comm comm_world) {
    parallel_mergesort(first, last, 0, 0, max_rank, comm_world);
    return;
}

void starter_child (int rank, int max_rank, MPI_Comm comm_world) {
    int group = 0;
    while ((1<<group) <= rank) ++group;
    
    MPI_Status status; int size;
    MPI_Probe(MPI_ANY_SOURCE, 666, comm_world, &status);
    
    MPI_Get_count(&status, MPI_INT, &size);
    
    int parent_rank = status.MPI_SOURCE;
    int* a = new int[size];
    MPI_Recv(a, size, MPI_INT, parent_rank, 666, comm_world, &status);
    
    parallel_mergesort(a, a + size, group, rank, max_rank, comm_world);

    MPI_Send(a, size, MPI_INT, parent_rank, 666, comm_world);
    
    delete[] a;
    return;
}


int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
    
    int comm; MPI_Comm_size(MPI_COMM_WORLD, &comm);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int max_rank = comm - 1;
    
    std::vector<int> sizes(5);
    sizes[0] = 100000;
    sizes[1] = 1000000;
    sizes[2] = 10000000;
    sizes[3] = 50000000;
    sizes[4] = 100000000;

    for(size_t s = 0; s < 5; ++s)
    {
        if (rank == 0) {
            int* a = new int[sizes[s]];
            
            for(size_t i=0; i < sizes[s]; ++i) {
                a[i] = rand() % MAX_Z + MIN_Z;
            }
            
            double start = MPI_Wtime();
            starter_root(a, a + sizes[s], max_rank, MPI_COMM_WORLD);
            double end = MPI_Wtime();
            std::cout << end - start << "," << sizes[s] << std::endl;
                        
            delete[] a;
        }
        else
        {
            starter_child(rank, max_rank, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}
