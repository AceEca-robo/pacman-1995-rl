#ifndef __dynamiel_h_ 
 #define __dynamiel_h_
 #include"element.h"
#include"gdynelem.h"
#include"usrinter.h"
#include"board.h"
#include"usrinter.h"

class Board;

class DynamicElement : public Element {		//from now on it is a sprite

protected:
int x,y;					//the sprites coordinates
DynamicGraphElement *g;		//pointer to the sprite's graphic element

public:
virtual void draw(); 				//for drawing the sprite 
void getxy(int*,int*);				//get it's coordinates
virtual GID_TYPE getgid();			//get it's grahical id
};

#endif

