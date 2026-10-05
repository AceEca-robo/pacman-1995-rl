#include"t90.h"
  
T90::T90() {
 
consfn();
pix(&pixmap,(char*)t90_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
}


