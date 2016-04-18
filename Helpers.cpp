#include <cstdio>
#include "Helpers.h"
#include "Globals.h"
#include <cstdlib>
#include <ctime>

//returns 0 - no error; 1 - if error occured
int LoadGridFromFile(const int width,const int height, int* grid, char* filename)
{
	FILE *file = fopen(filename, "r");
	

	char readChar = getc(file);
	int i = 0, j = 0;
	while (readChar != EOF)
	{
		if (readChar == '0' || readChar == '1')
		{
			grid[i*width + j] = readChar - '0';
		}
		else if (readChar == ' ')
		{
			j++;
		}
		else if (readChar == '\n')
		{
			i++;
			if (j >= width) //too many values in a row
				return 1;

			j = 0; //reset column counting
		}
		else
		{
			return 1;
		}
		readChar = getc(file);
	}
	if (i >= height) //too many rows
		return 1;
	
	fclose(file);
	return 0;
}

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

void randomMap(int *i_cells, int width, int height)
{
	srand(time(NULL));
	for (int i = 0; i < width*height; i++)
	{
		i_cells[i] = rand() % 100 >= (100 - SCREEN_COVERAGE) ? 1 : 0;
	}
}

void usage(char* name)
{
	fprintf(stderr, "%s [-w worldW worldH [-f width height filename]]\n", name);
	fprintf(stderr, "worldW - world width, worldH - world height \n");
	fprintf(stderr, "width, height - dimensions of initial the grid (not of world!)\n");
	fprintf(stderr, "filename - name of provided initial part of world\n");
	exit(EXIT_FAILURE);
}

