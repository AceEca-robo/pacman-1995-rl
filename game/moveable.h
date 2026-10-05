#ifndef __moveable_h_ 
#define __moveable_h_
#include"gamedata.h"
#include"dynamiel.h"
#include"gamedata.h"

class Moveables : public DynamicElement {
 protected:
  UserInterface* u;		//pointer to the userinterface
Board* b;			//pointer to the board

public:
virtual void die(Gamedata*)=0;	//virtual die function
        
};

#endif
