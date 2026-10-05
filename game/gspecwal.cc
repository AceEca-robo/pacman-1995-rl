#include"gspecwal.h"
  
G_SpecialWall::G_SpecialWall() {
 
consfn();
pix(&pixmap,(char*)specwall_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);

};


