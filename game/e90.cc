#include"e90.h"
  
E90::E90() {
 
consfn();
pix(&pixmap,(char*)e90_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
}
