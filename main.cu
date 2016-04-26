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
	Max dimensions of the world: 6000 x 5500
	Parameters: Intel Core i5-2500K 3.30GHz
				8GB RAM
				NVIDIA GeForce GTX 560 Ti 
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
#include "Helpers.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int isPaused = 0;

unsigned int world_width;
unsigned int world_height;

int *cells;
int *d_ocells;
int *d_icells;
struct uchar4 *d_dst;

#define BUFFER_DATA(i) ((char *)0 + i)

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
//Source image on the host side
uchar4 *h_Src;
//Size of displayed image
int imageW, imageH;

int offsetX = 0, offsetY = 0; //where camera starts rendering
float scale = 1;
float alreadyScaled = 1.2;
float slow = 1;
void RunGameOfLifeKernel(int *i_cells, int *o_cells, int width, int height, uchar4* dst);
float milliseconds =1.0;

////////////////////////////////////////////////////////////////////////////////
//OPENGL FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
void displayFunc(void)
{
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_dst = NULL;

	cudaGLMapBufferObject((void**)&d_dst, gl_PBO);

	cudaEventRecord(start);
	RunGameOfLifeKernel(d_icells, d_ocells, world_width, world_height, d_dst);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaGLUnmapBufferObject(gl_PBO);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));
	glBegin(GL_TRIANGLES);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(2.0f, 0.0f);
	glVertex2f(3.0f, -1.0f);
	glTexCoord2f(0.0f, 2.0f);
	glVertex2f(-1.0f, 3.0f);
	glEnd();

	glScalef(scale, scale, 1);
	scale = 1;
	glutSwapBuffers();
} // displayFunc

void idleFunc()
{
	glutPostRedisplay();
	int toSleep = (1000 - (int)milliseconds) / FPS*slow;
	int fps = 1000 / toSleep;
	char buff[50];
	sprintf(buff, "FPS: %i; X: %i; Y: %i; Slow: %.2f", fps, offsetX, offsetY, slow);
	glutSetWindowTitle( buff);
	std::chrono::milliseconds dur(toSleep);
	std::this_thread::sleep_for(dur);
}
void closeFunc()
{
	gpuErrchk(cudaFree(d_ocells));
	gpuErrchk(cudaFree(d_icells));
	free(h_Src);
	free(cells);
	printf("Closed\n");
}
void keyboardFunc(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'p':
		isPaused = isPaused == 0 ? 1 : 0;
		break;
	case 'w':
			offsetY = (offsetY += STEP_SIZE)%world_width;
		break;
	case 's':
		offsetY = (offsetY -= STEP_SIZE) % world_width;
		break;
	case 'a':
			offsetX = (offsetX -= STEP_SIZE)%world_height;
		break;
	case 'd':	
		offsetX = (offsetX += STEP_SIZE) % world_height;
		break;
	case 'm':
		slow += 0.1;
		break;
	case 'n':
		if (slow > 0.2)
		{
			slow -= 0.1;
		}
		
		break;
	//case 'z':
	//	scale = 1.1;
	//	alreadyScaled *= 1.1;
	//	break;
	//case 'x':
	//	alreadyScaled *= 0.9;
	//	scale = 0.9;
			
	default:
		break;
	}
}
bool initGL(int *argc, char **argv)
{
	imageH = MAX_WINDOW_HEIGHT;
	imageW = MAX_WINDOW_WIDTH;
	if (world_height <= MAX_WINDOW_HEIGHT)
		imageH = world_height;
	if (world_width <= MAX_WINDOW_WIDTH)
		imageW = world_width;

	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(imageW, imageH);
	glutCreateWindow(argv[0]);
	printf("Loading extensions: %s\n", (char*)glewGetErrorString(glewInit()));
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
	glutCloseFunc(closeFunc);
	glutKeyboardFunc(keyboardFunc);
	return true;
}//initgl

//end opengl functions


///////////////////////////////////////////////////////////////////////////////
//GPU FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

__device__ int CountAliveCells(int *i_cells, int idx, int width, int height)
{
	int alive = 0;

	int posY = floorf(idx / width);
	int posX = idx % width;

	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			int currPosX = (posX + i) % width;
			int currPosY = (posY + j) % height;

			if (currPosX < 0)
			{
				currPosX = width + currPosX;
			}
			if (currPosY < 0)
			{
				currPosY = height + currPosY;
			}

			int neigh = currPosY * width + currPosX;

			if (i == 0 && j == 0)
				continue;
			if (i_cells[neigh] == 1)
				alive++;
		}
	}
	return alive;
}
__global__ void UpdateGrid(int offX, int offY ,int *i_cells, struct uchar4 *dst, int width, int height, int worldW, int worldH)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //position in visible grid
	int posX =offX + idx%width;
	int posY =offY + idx / width;


	int cellIdx = posX*worldW + posY; //calc position of cell in grid
	if ( idx < width*height)
	{
		//assign color
		dst[idx].x = i_cells[cellIdx] * 255;
		dst[idx].y = i_cells[cellIdx] * 255;
		dst[idx].z = i_cells[cellIdx] * 255;
	}
}

__global__ void CalcNextGeneration(int *i_cells, int *o_cells, int width, int height)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= width*height)
		return;

	int neighCount = CountAliveCells(i_cells, idx, width, height);


	if (neighCount == 3 ||
		(neighCount == 2 && i_cells[idx] == 1))
		o_cells[idx] = 1;
	else
		o_cells[idx] = 0;

	__syncthreads();

}
__global__ void SyncCells(int *i_cells, int *o_cells, int size)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
		return;
	i_cells[idx] = o_cells[idx];
}
//end gpu functions

void RunGameOfLifeKernel(int *i_cells, int *o_cells, int worldW, int worldH, uchar4* dst)
{
	int worldSize = worldW*worldH;
	dim3 threads(THREADS_PER_BLOCK, 1, 1);
	dim3 blocks(ceil(((float)worldSize) / (float)(THREADS_PER_BLOCK)), 1, 1);
	if (!isPaused)
	{
		CalcNextGeneration << <blocks, threads >> >(i_cells, o_cells, worldW, worldH);
		gpuErrchk(cudaGetLastError());

		SyncCells << <blocks, threads >> >(i_cells, o_cells, worldSize); //sync cells betweeen threads
		gpuErrchk(cudaGetLastError());
	}
	dim3 threadsGrid(THREADS_PER_BLOCK, 1, 1);
	dim3 blocksGrid(ceil(((float)imageW*imageH) / (float)(THREADS_PER_BLOCK)), 1, 1);
	UpdateGrid << <blocksGrid, threadsGrid >> >(offsetX, offsetY, i_cells, dst, imageW, imageH, worldW, worldH); //draw stuff
	gpuErrchk(cudaGetLastError());
}

void initWorld(int gridWidth, int gridHeight, char* filename)
{
	int *grid = (int*)malloc(gridWidth*gridHeight * sizeof(int));

	if (LoadGridFromFile(gridWidth, gridHeight, grid, filename) == 1)
	{
		printf("Error occured!\n Not proper data in file: %s", filename);
		exit(EXIT_FAILURE);
	}

	cells = (int*)calloc(world_width*world_height, sizeof(int));
	//printCells(grid, gridWidth, gridHeight);
	int offsetX = 0;// world_height / 2 - gridHeight / 2;
	int offsetY = 0;// world_width / 2 - gridWidth / 2;

	for (int i = 0; i < gridHeight; i++)
	{
		for (int j = 0; j < gridWidth; j++)
		{
			int posX = (offsetX + i) % world_width;
			int posY = (offsetY + j) % world_height;
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
	world_width = 500;
	world_height = 500;
	int gridWidth;
	int gridHeight;
	switch (argc)
	{ 
	case 4: //random map with provided grid dimensions
		if (strcmp(argv[1], "-w")==0)
		{
			world_width = atoi(argv[2]);
			world_height = atoi(argv[3]);
			if (world_width < 5 || world_height < 5)
			{
				usage(argv[0]);
			}
		}
		else
		{
			usage(argv[0]);
		}
	case 1: //init random map whether with initial size or with default (1000x1000)
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
	printf("Cells grid created...\n");
	gpuErrchk(cudaMalloc(&d_icells, size * sizeof(int)));
	gpuErrchk(cudaMalloc(&d_ocells, size * sizeof(int)));
	
	gpuErrchk(cudaMemcpy(d_icells, cells, size*sizeof(int), cudaMemcpyHostToDevice));

	initGL(&argc, argv);
	printf("\nStarted\n");	
	printf("Controls:\n");
	printf("p - to pause/resume\n");
	printf("w/s/a/d - to move on the map\n");
	printf("m - slow animation\n");
	printf("n - speed up animation\n");
	//printf("z - zoom out\n");
	//printf("x - zoom in\n");

	glutMainLoop();
	printf("Ended\n");
}