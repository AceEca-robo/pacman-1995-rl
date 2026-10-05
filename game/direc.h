#ifndef __direc_h_
#define __direc_h_

//defines possible directions
typedef enum {up,down,left,right,none,still} direction;

//get next coordinates with a certain direction
void next(int*,int*,direction,int,int);

#endif
