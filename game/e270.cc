#include"e270.h"
  
E270::E270() {
 
consfn();
pix(&pixmap,(char*)e270_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
}

