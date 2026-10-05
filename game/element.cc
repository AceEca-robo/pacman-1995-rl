#include"element.h"

void Element::draw(int xx,int yy) {	//for drawing the element
g->draw(xx,yy);
}

void Element::gset(GraphElement *gg) {//set the pointer to the graphic element
g=gg;
}
