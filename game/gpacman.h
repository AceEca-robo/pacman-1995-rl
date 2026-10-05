#ifndef __gpacman_h_ 
#define __gpacman_h_
#include"gmoveabl.h"
#include"usrinter.h"
#include"colour.h"
#include"direc.h"

class G_Pacman : public G_Moveables {

PIXMAPTYPE pix0,pixl,pixr,pixu,pixd;	//pixmaps for mouth shut, and mouth
					//open in left right up down direction

direction look;				//direction to look
int mouth;				//mouth open?

public:

G_Pacman();				//constructor
~G_Pacman();				//destructor
void draw(int,int,direction,direction);	//draw pacman x,y direction and try_d
GID_TYPE getgid(direction,direction);

};

#endif

