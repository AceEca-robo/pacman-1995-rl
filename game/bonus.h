#ifndef __bonus_h_ 
#define __bonus_h_
#include"gamedata.h"
#include"board.h"
#include"usrinter.h"
#include"unmovele.h"
#include"gamedata.h"
#include"gbonus.h"

class Bonus : public UnmoveableElement {
protected:
int lifetime; 			//the lifetime of the bonus
            
public: 
virtual void start(); 		//reset the bonus
void getxy(int *,int*); 	//get it's coordinates
virtual void eat(Gamedata*)=0;	//a virtual for when pacman eats it
int code(); 			//code to be done while it exists
virtual ~Bonus()=0;		//destructor
Bonus();			//constructor

};

#endif 
