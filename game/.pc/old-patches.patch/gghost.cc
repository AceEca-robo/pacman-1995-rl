#include"gghost.h"
  
G_Ghost::G_Ghost(COLOURTYPE farge) {

consfn();
pix(&pixmapn,ghost_bits,farge,Colour::MYBACKGROUND);
pix(&pixmapr,ghost_bits,Colour::RUNGHOSTCOLOUR,Colour::MYBACKGROUND);
pixmap=pixmapn;

};

G_Ghost::~G_Ghost(void) {
#ifdef XWIN
if (pixmap!=pixmapr) XFreePixmap(display,pixmapr);
if (pixmap!=pixmapn) XFreePixmap(display,pixmapn);
#elif defined MSWIN
if (pixmap!=pixmapr) DeleteObject(pixmapr);
if (pixmap!=pixmapn) DeleteObject(pixmapn);
#endif
}

void G_Ghost::draw(int x,int y,int run) {//draw at x,y with running boolean
if (!run) pixmap=pixmapn; else pixmap=pixmapr;
GraphElement::draw(x,y);
}

