#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

//background, buf1, mask, output,
__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];	
		}
	}
}

__global__ void CalculateFixed(
    // background, target, mask, fixed, wb, hb, wt, ht, oy, ox
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt ) {      //white               
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			
			float tup0, tup1, tup2, tdown0, tdown1, tdown2, tleft0, tleft1, tleft2, tright0, tright1, tright2;
		
			if(xt==0){
				tleft0 = 0; tleft1 = 0; tleft2 = 0;
			}
			else{
				tleft0 = target[(curt-1)*3+0]; tleft1 = target[(curt-1)*3+1]; tleft2 = target[(curt-1)*3+2]; 
			}
			
			if(xt==wt-1){
				tright0 = 0; tright1 = 0; tright2 = 0;
			}
			else{
				tright0 = target[(curt+1)*3+0]; tright1 = target[(curt+1)*3+1]; tright2 = target[(curt+1)*3+2]; 
			}
			
			if(yt==0){
				tup0 = 0; tup1 = 0; tup2 = 0;
			}
			else{
				tup0 = target[(curt-wt)*3+0]; tup1 = target[(curt-wt)*3+1]; tup2 = target[(curt-wt)*3+2];
			}
			
			if(yt==ht-1){
				tdown0 = 0; tdown1 = 0; tdown2 = 0;
			}
			else{
				tdown0 = target[(curt+wt)*3+0]; tdown1 = target[(curt+wt)*3+1]; tdown2 = target[(curt+wt)*3+2];
			}
			
			if(xt==0 || xt==wt-1 || yt==0 || yt==ht-1){
				fixed[curt*3+0] = 3*target[curt*3+0] - (tleft0+tright0+tup0+tdown0);
				fixed[curt*3+1] = 3*target[curt*3+1] - (tleft1+tright1+tup1+tdown1);
				fixed[curt*3+2] = 3*target[curt*3+2] - (tleft2+tright2+tup2+tdown2);		
			}
			else{
				fixed[curt*3+0] = 4*target[curt*3+0] - (tleft0+tright0+tup0+tdown0);
				fixed[curt*3+1] = 4*target[curt*3+1] - (tleft1+tright1+tup1+tdown1);
				fixed[curt*3+2] = 4*target[curt*3+2] - (tleft2+tright2+tup2+tdown2);		
			}
				
			// Nb Sb Wb Eb may have black
			if( yt==0 || yt>0 && mask[curt-wt] < 127.0f){    //up is black
				fixed[curt*3+0] += background[(curb-wb)*3+0];
				fixed[curt*3+1] += background[(curb-wb)*3+1];
				fixed[curt*3+2] += background[(curb-wb)*3+2];
			}
			if( yt==ht-1 || yt<ht-1 && mask[curt+wt] < 127.0f){    //down is black
				fixed[curt*3+0] += background[(curb+wb)*3+0];
				fixed[curt*3+1] += background[(curb+wb)*3+1];
				fixed[curt*3+2] += background[(curb+wb)*3+2];
			}
			if( xt==0 || xt>0 && mask[curt-1] < 127.0f){     //left is black
				fixed[curt*3+0] += background[(curb-1)*3+0];
				fixed[curt*3+1] += background[(curb-1)*3+1];
				fixed[curt*3+2] += background[(curb-1)*3+2];
			}
			if( xt==wt-1 || xt<wt-1 && mask[curt+1] < 127.0f){     //right is black
				fixed[curt*3+0] += background[(curb+1)*3+0];
				fixed[curt*3+1] += background[(curb+1)*3+1];
				fixed[curt*3+2] += background[(curb+1)*3+2];
			}
			
		}
	}
}

__global__ void PoissonImageCloningIteration(
    // fixed, mask, buf1, buf2, wt, ht
	// fixed, mask, buf2, buf1, wt, ht
	float *fixed,
	const float *mask,	
	float *input,
	float *output,
	const int wt, const int ht
){
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		output[curt*3+0] = 0;
		output[curt*3+1] = 0;
		output[curt*3+2] = 0;
		// if neighbors has white 
		if( yt-1>=0 && mask[curt-wt] > 127.0f){   //top is white
			output[curt*3+0] += input[(curt-wt)*3+0]*1/4;
			output[curt*3+1] += input[(curt-wt)*3+1]*1/4;
			output[curt*3+2] += input[(curt-wt)*3+2]*1/4;
		}
		if( yt+1<ht && mask[curt+wt] > 127.0f){   //down is white
			output[curt*3+0] += input[(curt+wt)*3+0]*1/4;
			output[curt*3+1] += input[(curt+wt)*3+1]*1/4;
			output[curt*3+2] += input[(curt+wt)*3+2]*1/4;
		}
		if( xt-1>=0 && mask[curt-1] > 127.0f){    //left is white
			output[curt*3+0] += input[(curt-1)*3+0]*1/4;
			output[curt*3+1] += input[(curt-1)*3+1]*1/4;
			output[curt*3+2] += input[(curt-1)*3+2]*1/4;
		}
		if( xt+1<wt && mask[curt+1] > 127.0f){    //right is white
			output[curt*3+0] += input[(curt+1)*3+0]*1/4;
			output[curt*3+1] += input[(curt+1)*3+1]*1/4;
			output[curt*3+2] += input[(curt+1)*3+2]*1/4;
		}
		output[curt*3+0] = output[curt*3+0] + fixed[curt*3+0]*1/4;
		output[curt*3+1] = output[curt*3+1] + fixed[curt*3+1]*1/4;
		output[curt*3+2] = output[curt*3+2] + fixed[curt*3+2]*1/4;
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox 
){
    // set up
    float *fixed, *buf1, *buf2;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf2, 3*wt*ht*sizeof(float));
	
	// initialize the iteration
    dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
    CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
    );
    cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);
	
	
	// iterate
    for (int i = 0; i < 10000; ++i) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
    }
	
	// copy the image back
    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
    );

    // clean up
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
	
	/*
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	*/
}
