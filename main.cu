////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#include<GL/glew.h>

#include "Globals.h"
#include <time.h>
#include <thread>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


int *cells;
int *d_ocells;
int *d_icells;
struct uchar4 *d_dst;

////////////////////////////////////////////////////////////////////////////////
//// constants
//const unsigned int window_width  = 32;
//const unsigned int window_height = 32;
//
//const unsigned int world_width = 32;
//const unsigned int world_height = 32;


////////////////////////////////////////////////////////////////////////////////
// GL functionality
//bool initGL(int *argc, char **argv);
//void display();

//TMP
void printCells(int* cells)
{
	for (int i = 0; i < world_width * world_height; i++)
	{
		if (i%world_width == 0)
			printf("\n");
		printf("%i ", cells[i]);
	}
}
void randomMap(int *i_cells, int width, int height)
{
	srand(time(NULL));
	for (int i = 0; i < width*height; i++)
	{
		i_cells[i] = rand()%100 >= 90 ? 1 : 0;
		if (i < 20)
			printf("%i, ", i_cells[i]);
	}
}

void RunGameOfLifeKernel(int *i_cells, int *o_cells, int width, int height, uchar4* dst);

////////////////////////////////////////////////////////////////////////////////
//OPENGL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

#define BUFFER_DATA(i) ((char *)0 + i)

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
//Source image on the host side
uchar4 *h_Src;

//Original image width and height
int imageW, imageH;

// Timer ID
unsigned int hTimer;

void displayFunc(void)
{
	if ((xdOff != 0.0) || (ydOff != 0.0)) {
		xOff += xdOff;
		yOff += ydOff;
	}
	if (dscale != 1.0) {
		scale *= dscale;
	}
	if (animationStep) {
		animationFrame -= animationStep;
	}
	d_dst = NULL;
	float difftime = 0;


	cudaGLMapBufferObject((void**)&d_dst, gl_PBO);

	RunGameOfLifeKernel(d_icells, d_ocells, world_width, world_height, d_dst);

	//while (timeEstimate < 1.0f / 60.0f)
	//busy waiting 
	//TODO: change it!
	unsigned long start = glutGet(GLUT_ELAPSED_TIME);
	//do
	//{
	//	long end = glutGet(GLUT_ELAPSED_TIME);
	//	difftime = (end - start) / 1000;
	//} while (difftime < 1 / FPS);

	cudaGLUnmapBufferObject(gl_PBO);
//#if RUN_TIMING
//	printf("GPU = %5.8f\n", 0.001f * cutGetTimerValue(hTimer));
//#endif
	
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
	glBegin(GL_TRIANGLES);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(2.0f, 0.0f);
	glVertex2f(3.0f, -1.0f);
	glTexCoord2f(0.0f, 2.0f);
	glVertex2f(-1.0f, 3.0f);
	glEnd();

	glutSwapBuffers();
} // displayFunc

void idleFunc()
{
	glutPostRedisplay();
	std::chrono::milliseconds dur(1000 / 15);  // About 30 fps
	std::this_thread::sleep_for(dur);
}

bool initGL(int *argc, char **argv)
{
	imageH = world_height;
	imageW = world_width;


	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	/*glutInitWindowSize(imageW, imageH);*/
	glutInitWindowSize(imageW, imageH);
	//glutInitWindowPosition(512 - imageW / 2, 384 - imageH / 2);
	glutCreateWindow(argv[0]);
	printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
	if (!glewIsSupported(
		"GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		)){
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}
	printf("OpenGL window created.\n");

	printf("Creating GL texture...\n");
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, imageW, imageH, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
	printf("Texture created.\n");

	printf("Creating PBO...\n");
	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, imageW * imageH * 4, h_Src, GL_STREAM_COPY);
	//While a PBO is registered to CUDA, it can't be used 
	//as the destination for OpenGL drawing calls.
	//But in our particular case OpenGL is only used 
	//to display the content of the PBO, specified by CUDA kernels,
	//so we need to register/unregister it only once.
	cudaGLRegisterBufferObject(gl_PBO);
	printf("PBO created.\n");

	glutDisplayFunc(displayFunc);
	glutIdleFunc(idleFunc);
	return true;
}

//end opengl functions




///////////////////////////////////////////////////////////////////////////////
//GPU FUNCTIONS
///////////////////////////////////////////////////////////////////////////////
__device__ int CountAliveCells(int *i_cells, int idx, int width, int height)
{
	int alive = 0;

	int posY = idx / width;
	int posX = idx - posY*width;

	//if (idx == 4)
	//	printf("Idx: 4; x = %i, y = %i;\n", posX, posY);

	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			//clean this up!!!
			int currPosX = ((posX + i) % width) >= 0 ? (posX + i) % width : width +( (posX + i) % width);
			int currPosY =( (posY + j) % height )>= 0 ? (posY + j) % height : height + ((posY + j) % height);
			//int neigh = (idx + i*(width) + j) % (width*height);

			int neigh = currPosY * width + currPosX;
			
			//if (idx == 4)
			//	printf("neighIdx: %i\n", neigh);

			if (i == 0 && j == 0)
				continue;
			if (i_cells[neigh] == 1)
				alive++;
		}
	}
	return alive;
}

__global__ void CalcNextGeneration(int *i_cells, int *o_cells, int width, int height, uchar4 *dst)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx > width*height)
		return;
	if (idx == 24)
	{
		printf("512 cell started...\n");
		printf("init: %i\n", i_cells[idx]);
		printf("Idx: %i;Neigh: %i; \n", idx, CountAliveCells(i_cells, idx, width, height));
	}

	if (CountAliveCells(i_cells, idx,width,height) == 3|| (CountAliveCells(i_cells, idx,width,height) == 2 && i_cells[idx]==1))
		o_cells[idx] = 1;
	else
		o_cells[idx] = 0;
	
	__syncthreads();
	i_cells[idx] = o_cells[idx];
	
	//assign color
	dst[idx].x = i_cells[idx] * 255;
	dst[idx].y = i_cells[idx] * 255;
	dst[idx].z = i_cells[idx] * 255;
	if (idx == 24)
		printf("color: %i\n", dst[idx].x);
}

//end gpu functions

void RunGameOfLifeKernel(int *i_cells, int *o_cells, int width, int height, uchar4* dst)
{
	int size = width*height;
	dim3 threads(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks(ceil(size / THREADS_PER_BLOCK), 1, 1);
	//printf("Kernel started\n");
	CalcNextGeneration<<<blocks, threads >>>(i_cells, o_cells, width, height, dst);
	gpuErrchk(cudaGetLastError());
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{	

	//int *d_icells;
	
	int size = world_height * world_width;


	//init
	cells = (int*)malloc(size* sizeof(int));
	randomMap(cells, world_width, world_height);
	//for (int i = 0; i < size; i++)
	//{
	//	cells[i]= 0;
	//}

	//cells[24] = 1;
	//cells[25] = 1;
	//cells[23] = 1;

	gpuErrchk(cudaMalloc(&d_icells, size * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_ocells, size*sizeof(int)));
	gpuErrchk(cudaMemcpy(d_icells, cells, size*sizeof(int), cudaMemcpyHostToDevice));
	initGL(&argc, argv);
	//printCells(cells);
	printf("\nStarted\n");	

	//CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	//CUT_SAFE_CALL(cutStartTimer(hTimer));
	glutMainLoop();
}



////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
//bool initGL(int *argc, char **argv)
//{
//	glutInit(argc, argv);
//	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//
//	glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
//	glClear(GL_COLOR_BUFFER_BIT);
//	//glMatrixMode(GL_MODELVIEW);
//	glutInitWindowSize(window_width, window_height);
//	glutCreateWindow("Hello World");
//
//	glutDisplayFunc(display);
//
//	
//	return 0;
//}
//
//void display()
//{
//	//TODO: generate vertex positions
//	//
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	//glColor3f(0.0f, 0.0f, 0.0f);
//	glDrawPixels(world_width, world_height, GL_RGB, GL_UNSIGNED_BYTE, easel);
//	
//	glutSwapBuffers();
//}


