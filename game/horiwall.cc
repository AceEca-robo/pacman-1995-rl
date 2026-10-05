#include"horiwall.h"  


HorizontalWall::HorizontalWall() {
 
consfn();
pix(&pixmap,(char*)horiwall_bits,Colour::WALLCOLOUR,Colour::MYBACKGROUND);
};
