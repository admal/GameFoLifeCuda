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
#include "LoadingFiles.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


unsigned int world_width;
unsigned int world_height;

int *cells;
int *d_ocells;
int *d_icells;
struct uchar4 *d_dst;

//TMP
void printCells(int* cells, int width, int height)
{
	int size = width*height;
	for (int i = 0; i < size; i++)
	{
		if (i%width == 0)
			printf("\n");
		printf("%i ", cells[i]);
	}
	printf("\n");
}
//end tmp
void randomMap(int *i_cells, int width, int height)
{
	srand(time(NULL));
	for (int i = 0; i < width*height; i++)
	{
		i_cells[i] = rand()%100 >= (100 -SCREEN_COVERAGE) ? 1 : 0;
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
//unsigned int hTimer;

void displayFunc(void)
{
	d_dst = NULL;

	cudaGLMapBufferObject((void**)&d_dst, gl_PBO);

	RunGameOfLifeKernel(d_icells, d_ocells, world_width, world_height, d_dst);

	cudaGLUnmapBufferObject(gl_PBO);

	//glPushMatrix();  //begin scaling
	//glScalef(10, 10, 10);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
	glBegin(GL_TRIANGLES);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(2.0f, 0.0f);
	glVertex2f(3.0f, -1.0f);
	glTexCoord2f(0.0f, 2.0f);
	glVertex2f(-1.0f, 3.0f);
	glEnd();
	//glPopMatrix();
	glutSwapBuffers();
} // displayFunc

void idleFunc()
{
	glutPostRedisplay();
	std::chrono::milliseconds dur(1000 / FPS);
	std::this_thread::sleep_for(dur);
}

bool initGL(int *argc, char **argv)
{
	imageH = world_height;
	imageW = world_width;


	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
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

	int posY = floorf(idx/width);
	//int posX = idx - posY*width;
	int posX = idx % width;

	//if (idx == 4)
	//	printf("Idx: 4; x = %i, y = %i;\n", posX, posY);

	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int currPosX = (posX + i) % width;
			int currPosY = (posY + j) % height;

			if (currPosX < 0)
			{
				currPosX = width + currPosX - 1;
			}
			if (currPosY < 0)
			{
				currPosY = height + currPosY - 1;
			}

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
	if (idx >= width*height)
		return;
	if (idx == 24)
	{
		printf("24 cell started...\n");
		printf("init: %i\n", i_cells[idx]);
		printf("Idx: %i;Neigh: %i; \n", idx, CountAliveCells(i_cells, idx, width, height));
	}

	if (CountAliveCells(i_cells, idx,width,height) == 3|| 
		(CountAliveCells(i_cells, idx,width,height) == 2 && i_cells[idx]==1))
		o_cells[idx] = 1;
	else
		o_cells[idx] = 0;
	
	__syncthreads();
	i_cells[idx] = o_cells[idx];
	
	//assign color
	dst[idx].x = i_cells[idx] * 255;
	dst[idx].y = i_cells[idx] * 255;
	dst[idx].z = i_cells[idx] * 255;
	//if (idx == 24)
	//	printf("color: %i\n", dst[idx].x);
}

//end gpu functions

void RunGameOfLifeKernel(int *i_cells, int *o_cells, int width, int height, uchar4* dst)
{
	int size = width*height;
	dim3 threads(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks(ceil(size / THREADS_PER_BLOCK), 1, 1);

	CalcNextGeneration<<<blocks, threads >>>(i_cells, o_cells, width, height, dst);
	gpuErrchk(cudaGetLastError());
}


void usage(char* name)
{
	fprintf(stderr, "%s [-w worldW worldH [-f width height filename]]\n", name);
	fprintf(stderr, "worldW - world width, worldH - world height \n");
	fprintf(stderr, "width, height - dimensions of initial the grid (not of world!)\n");
	fprintf(stderr, "filename - name of provided initial part of world\n");
	exit(EXIT_FAILURE);
}

void initWorld(int gridWidth, int gridHeight, char* filename)
{
	int *grid = (int*)malloc(gridWidth*gridHeight * sizeof(int));

	if ( LoadGridFromFile(gridWidth, gridHeight, grid, filename) == 1)
	{
		printf("Error occured!\n Not proper data in file: %s", filename);
		exit(EXIT_FAILURE);
	}


	cells = (int*)calloc(world_width*world_height, sizeof(int));
	printCells(grid, gridWidth,gridHeight);
	int offsetX = 0;// world_height / 2 - gridHeight / 2;
	int offsetY = 0;// world_width / 2 - gridWidth / 2;

	for (int i = 0; i < gridWidth; i++)
	{
		for (int j = 0; j < gridHeight; j++)
		{
			int posX = offsetX + i;
			int posY = offsetY + j;
			cells[posX*world_width + posY] = grid[i*gridWidth + j];
		}
	}

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{	
	int size;
	world_width = 1000;
	world_height = 1000;
	int gridWidth;
	int gridHeight;
	printf("argc = %i\n", argc);
	switch (argc)
	{ 
	case 4: //random map with provided grid
		printf("argv[1] = %s\n", argv[1]);
		if (strcmp(argv[1], "-w")==0)
		{
			world_width = atoi(argv[2]);
			world_height = atoi(argv[3]);

			if (world_width < 5 || world_height < 5)
			{
				printf("Error line 356\n");
				usage(argv[0]);
			}
		}
		else
		{
			printf("Error line 362\n");
			usage(argv[0]);
		}
	case 1: //init random map whether with initial size or with default
		size = world_height * world_width;
		cells = (int*)malloc(size* sizeof(int));
		randomMap(cells, world_width, world_height);
		break;
	case 8:
		world_width = atoi(argv[2]);
		world_height = atoi(argv[3]);
		gridWidth = atoi(argv[5]);
		gridHeight = atoi(argv[6]);
		size = world_height * world_width;
		initWorld(gridWidth, gridHeight, argv[7]);
		break;
	default:
		usage(argv[0]);
		break;
	}

	gpuErrchk(cudaMalloc(&d_icells, size * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_ocells, size*sizeof(int)));
	gpuErrchk(cudaMemcpy(d_icells, cells, size*sizeof(int), cudaMemcpyHostToDevice));
	initGL(&argc, argv);
	printf("\nStarted\n");	

	glutMainLoop();
}