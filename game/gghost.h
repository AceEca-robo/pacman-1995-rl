#ifndef __gghost_h_ 
#define __gghost_h_
#include"gmoveabl.h"

class G_Ghost : public G_Moveables {

PIXMAPTYPE pixmapn,pixmapr;	//pixmaps for normal and running colours
          
public:

G_Ghost(COLOURTYPE);		//normal colours
~G_Ghost();
void draw(int,int,int);		//draw at x,y with running boolean
          

};

#endif
