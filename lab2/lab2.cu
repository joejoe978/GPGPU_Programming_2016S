#include "lab2.h"
#include "cmath"
#include "stdlib.h"
#include <iostream>
#include "PerlinNoise.h"
#include <random>
#include <algorithm>
#include <time.h>

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;

PerlinNoise::PerlinNoise() {
	
	// Initialize the permutation vector with the reference values
	p = {
		151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
		8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
		35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,
		134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
		55,46,245,40,244,102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
		18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
		250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
		189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 
		43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,
		97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,
		107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
		138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };
	// Duplicate the permutation vector
	p.insert(p.end(), p.begin(), p.end());
}

PerlinNoise::PerlinNoise(unsigned int seed) {
	p.resize(256);

	// Fill p with values from 0 to 255
	int indx = 0;
	std::generate(p.begin(), p.end(), [&indx] {return indx++;});
	
	// Initialize a random engine with seed
	std::default_random_engine engine(seed);

	// Shuffle  using the above random engine
	std::shuffle(p.begin(), p.end(), engine);

	// Duplicate the permutation vector
	p.insert(p.end(), p.begin(), p.end());
}

double PerlinNoise::noise(double x, double y, double z) {
	// Find the unit cube that contains the point
	int X = (int) floor(x) & 255;
	int Y = (int) floor(y) & 255;
	int Z = (int) floor(z) & 255;

	// Find relative x, y,z of point in cube
	x -= floor(x);
	y -= floor(y);
	z -= floor(z);

	// Compute fade curves for each of x, y, z
	double u = fade(x);
	double v = fade(y);
	double w = fade(z);

	// Hash coordinates of the 8 cube corners
	int A = p[X] + Y;
	int AA = p[A] + Z;
	int AB = p[A + 1] + Z;
	int B = p[X + 1] + Y;
	int BA = p[B] + Z;
	int BB = p[B + 1] + Z;

	// Add blended results from 8 corners of cube
	double res = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x-1, y, z)), lerp(u, grad(p[AB], x, y-1, z), grad(p[BB], x-1, y-1, z))),	lerp(v, lerp(u, grad(p[AA+1], x, y, z-1), grad(p[BA+1], x-1, y, z-1)), lerp(u, grad(p[AB+1], x, y-1, z-1),	grad(p[BB+1], x-1, y-1, z-1))));
	return (res + 1.0)/2.0;
}

double PerlinNoise::fade(double t) { 
	return t * t * t * (t * (t * 6 - 15) + 10);
}

double PerlinNoise::lerp(double t, double a, double b) { 
	return a + t * (b - a); 
}

double PerlinNoise::grad(int hash, double x, double y, double z) {
	int h = hash & 15;
	// Convert lower 4 bits of hash into 12 gradient directions
	double u = h < 8 ? x : y,
		   v = h < 4 ? y : h == 12 || h == 14 ? x : z;
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
	unsigned int seed = rand() % 255;
	PerlinNoise pn(seed);
	
	cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	int i, j ;
	
	for(i=0;i<W*H/2;i++){
	 	// j = i % 256; // purple and green 
	 	// j = i % 128; // purple and green 
		
		double x = (double)i/((double)W);
		double y = (double)i/((double)H);
		double n = 20 * pn.noise(x,y,0.8);
		n = n - floor(n);
		//double n = pn.noise(10 * i, 10 * i, 0.8);
		cudaMemset(yuv+W*H+i, floor(n*255), 1);	
	}
	
	//cudaMemset(yuv+W*H, n, W*H/2); // 從綠變灰變粉紅
	//cudaMemset(yuv+W*H, 128, W*H/2);	// 灰階
	++(impl->t);
}
