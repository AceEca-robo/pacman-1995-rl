#include"vertwall.h"
  

VerticalWall::VerticalWall() {
 
consfn();
pix(&pixmap,(char*)vertwall_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
};
