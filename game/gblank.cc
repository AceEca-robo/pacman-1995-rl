#include"gblank.h"

G_Blank::G_Blank() {
 
consfn();
pix(&pixmap,(char*)blank_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
}


