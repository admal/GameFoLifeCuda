#include <cstdio>
#include "LoadingFiles.h"

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


	//for (int i = 0; i < width; i++)
	//{
	//	for (int j = 0; j < height; j++)
	//	{
	//		char readChar = getc(file);
	//		if (readChar == EOF)
	//			return 1;
	//		if (readChar == '0' || readChar == '1')
	//		{
	//			grid[i*width + j] = readChar - '0';
	//		}
	//		else if (readChar == ' ')
	//		{
	//			j--;
	//		}
	//		else if (readChar == '\n')
	//		{
	//			i--;
	//		}
	//		else
	//			return 1;
	//	}
	//}
	//return 0;
}
