#include"vertwall.h"
  

VerticalWall::VerticalWall() {
 
consfn();
pix(&pixmap,vertwall_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
};
