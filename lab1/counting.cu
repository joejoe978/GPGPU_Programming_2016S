#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
		
struct is_one{
    __host__ __device__
	bool operator()(int x){
		return x == 1;
	}
};

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


__global__ void init(const char *Gtext, int *Gpos, int Gtext_size, int *lastpos) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < Gtext_size){	
		//initialize	
		if(Gtext[idx]== '\n'){
			Gpos[idx] = 0;
			lastpos[idx] = 0;
		}
		else{
			Gpos[idx] = 1;
			lastpos[idx] = 1;
		}		
	}
}

__global__ void posParallel(int *Gpos, int Gtext_size, int *lastpos, int i, int j) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < Gtext_size){	
		if(idx > 0 && (idx - j >= 0))
			if(lastpos[idx] != 0 && (lastpos[idx-1] == lastpos[idx]))
				Gpos[idx] += lastpos[idx-j] ;
	}
}

__global__ void copy(int *Gpos, int Gtext_size, int *lastpos, int j) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < Gtext_size){	
		if(idx > 0 && (idx - j >= 0))
			lastpos[idx] = Gpos[idx];		
		
	}
}


void CountPosition(const char *text, int *pos, int text_size)
{	
	int blocksize = 512, i, j; 
	int *lastpos;
	size_t poslen = text_size * sizeof(int);
	
	cudaMalloc((void **) &lastpos, poslen);
    int nblock = text_size/blocksize + (text_size % blocksize == 0 ? 0 : 1); 

	init<<<nblock,blocksize>>>(text, pos, text_size, lastpos);  // initialize
	cudaDeviceSynchronize();
	
	for(i=0;i<9;i++){
		j = (1 << i);
		posParallel<<<nblock,blocksize>>>(pos, text_size, lastpos, i, j); 
		cudaDeviceSynchronize();	
		if(i != 8){
			copy<<<nblock,blocksize>>>(pos, text_size, lastpos, j); 
			cudaDeviceSynchronize();		
		}
	}	
	cudaFree(lastpos);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;   // number of heads
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO, head is answer.
	
	thrust::equal_to<int> equal;
	
	thrust::fill(cumsum_d, cumsum_d+text_size,1);  					  // buffer2 = 1111111111
	thrust::transform(pos_d,pos_d+text_size,cumsum_d,flag_d,equal);   // buffer = 01000100100
	nhead = thrust::reduce(flag_d, flag_d+text_size); 				  // get nhead
	//printf("nhead: %d \n", nhead);
	thrust::exclusive_scan(cumsum_d, cumsum_d+text_size, cumsum_d);   // buffer2 = 0123456789
	thrust::copy_if(cumsum_d, cumsum_d+text_size, flag_d, head_d, is_one());  // get heads
		
	cudaFree(buffer);
	return nhead;  
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
