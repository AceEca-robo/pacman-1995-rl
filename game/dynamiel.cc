#include"dynamiel.h"

void DynamicElement::getxy(int *xx,int *yy) {	//get it's coordinates
 *xx=x; *yy=y; }

void DynamicElement::draw() {			//for drawing the sprite
g->draw(x,y);
}

GID_TYPE DynamicElement::getgid() {		//get it's grahical id
return g->getgid();
}

