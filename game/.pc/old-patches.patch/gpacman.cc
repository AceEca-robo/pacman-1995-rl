#include"gpacman.h"
#include"pac.h"
  
G_Pacman::G_Pacman() {

consfn();
pix(&pix0,pacman0_bits,Colour::PACMANCOLOUR,Colour::MYBACKGROUND);
pix(&pixl,pacmanleft_bits,Colour::PACMANCOLOUR,Colour::MYBACKGROUND);
pix(&pixr,pacmanright_bits,Colour::PACMANCOLOUR,Colour::MYBACKGROUND);
pix(&pixu,pacmanup_bits,Colour::PACMANCOLOUR,Colour::MYBACKGROUND);
pix(&pixd,pacmandown_bits,Colour::PACMANCOLOUR,Colour::MYBACKGROUND);
pixmap=pixr; 
look=right;
mouth=1;
};
 
G_Pacman::~G_Pacman(void) {
#ifdef XWIN
if (pixmap!=pix0) XFreePixmap(display,pix0);
if (pixmap!=pixl) XFreePixmap(display,pixl);
if (pixmap!=pixr) XFreePixmap(display,pixr);
if (pixmap!=pixd) XFreePixmap(display,pixd);
if (pixmap!=pixu) XFreePixmap(display,pixu);
#elif defined MSWIN
if (pixmap!=pix0) DeleteObject(pix0);
if (pixmap!=pixl) DeleteObject(pixl);
if (pixmap!=pixr) DeleteObject(pixr);
if (pixmap!=pixd) DeleteObject(pixd);
if (pixmap!=pixu) DeleteObject(pixu);
#endif
}

void G_Pacman::draw(int x,int y,direction d,direction try_d) { 
if (d==still) mouth=1; 
if (try_d!=still) look=try_d;
//if (d==still) mouth=1; else look=d;
if (mouth) {
 switch (look) {
  case up: pixmap=pixu; break;
  case down: pixmap=pixd; break;
  case left: pixmap=pixl; break;
  case right: pixmap=pixr; break;
	 default: pacwarning("Gpac");
	 }
  } else pixmap=pix0;
mouth=1-mouth;			//if moving, mouth opens/closing alternating
GraphElement::draw(x,y);
}

//returns graphical id according whether mouth is open or which direction
GID_TYPE G_Pacman::getgid(direction d,direction try_d) {
PIXMAPTYPE pixmap;
           
if (d==still) mouth=1; 
if (try_d!=still) look=try_d;
//if (d==still) mouth=1; else look=d;
if (mouth) {
 switch (look) {
  case up: pixmap=pixu; break;
  case down: pixmap=pixd; break;
  case left: pixmap=pixl; break;
  case right: pixmap=pixr; break;
	 default: pacwarning("Gpac");
	 }
  } else pixmap=pix0;
return pixmap;
}

