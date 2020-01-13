#include <vector>
#include <iostream>
#include <algorithm>
#include <omp.h>

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
void parallel_mergesort(RandomIt first, RandomIt last, int threads) {
    if (threads == 1)
    {
        mergesort(first, last);
    }
    else
    {
       size_t diff = last - first;
       #pragma omp parallel sections num_threads(2)
       {
            #pragma omp section
            {
                parallel_mergesort(first, first + diff/2, threads/2);
            }
            #pragma omp section
            {
                parallel_mergesort(first + diff/2, last, threads - threads/2);
            }
       }
       std::inplace_merge(first, first + diff/2, last);
    }
}


int main() {
	omp_set_nested(1);

    std::vector<double> times_parallel(10);
    
    std::vector<int> sizes(5);
    sizes[0] = 100000;
    sizes[1] = 1000000;
    sizes[2] = 10000000;
    sizes[3] = 50000000;
    sizes[4] = 100000000;
    
    for(size_t s = 0; s < 5; ++s)
    {
        std::vector<int> a(sizes[s]);
        
        for(int th = 0; th <= 7; ++th)
        {
            
            int threads = (1<<th);
                    
            
            for(int j = 0; j < 10; ++j) {
                
                for(size_t i = 0; i < sizes[s]; i++) {
                    a[i] = rand() % MAX_Z + MIN_Z;
                }
                
                double start_parallel = omp_get_wtime();
                parallel_mergesort(a.begin(), a.end(), threads);
                double end_parallel = omp_get_wtime();
                
                times_parallel[j] = end_parallel - start_parallel;
            }
            double average_parallel = 0.0;
            
            for(int j = 0; j < 10; ++j) {
                average_parallel += times_parallel[j];
            }
            
            average_parallel /= 10;
            
            std::cout << threads << "," << average_parallel << "," << sizes[s] << std::endl;
        }
    }
    
    return 0;
}
