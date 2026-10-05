#include"direc.h"

//get next coordinates going in a certain direction
void next(int* x,int* y,direction d,int xx,int yy) {
*x=xx;
*y=yy;
switch (d) {
 case none: break;
case still: break;
case up: (*y)--; break;
case down: (*y)++; break;
case left: (*x)--; break;
case right: (*x)++; break;
	  }}
