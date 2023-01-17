#include <stdio.h>
#include <omp.h>



int main(int argc, char* argv[])
{
    #pragma omp target teams parallel \
        num_teams(108) thread_limit(64)
    {
        int globalId = omp_get_team_num() *  omp_get_num_threads() + omp_get_thread_num();

         
        printf("Hello world! I'm thread %d out of %d in team %d. My global thread id is %d out of %d\n", 
        omp_get_thread_num(), 
        omp_get_num_threads(), 
        omp_get_team_num(),
        globalId, 
        omp_get_num_threads() * omp_get_num_teams());

        if( globalId == 100 )
        {
            int* a = (int*) 0x10000; 
            *a = 0;
        }
    }

    return 0;
}