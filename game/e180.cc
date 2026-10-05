#include"e180.h"
  
E180::E180() {
 
consfn();
pix(&pixmap,(char*)e180_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
}

