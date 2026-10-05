#include"gblank.h"

G_Blank::G_Blank() {
 
consfn();
pix(&pixmap,blank_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
}


