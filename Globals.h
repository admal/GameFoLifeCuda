#ifndef _Globals_h_
#define _Globals_h_

#define MAX_ITER 20
#define THREADS_PER_BLOCK 900
#define FPS 5
#define SCREEN_COVERAGE 10 //percent of screen covered by living cells
// constants
//extern const unsigned int world_width;
//extern const unsigned int world_height;

// Starting position and scale
extern double xOff;
extern double yOff;
extern double scale;
// Starting stationary position and scale motion
extern double xdOff;
extern double ydOff;
extern double dscale;

// Starting animation frame
extern int animationFrame;
extern int animationStep;

//tmp

#endif
