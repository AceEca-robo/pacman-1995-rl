#include"bonus.h"
#include"pac.h"

void Bonus::getxy(int *xx,int *yy) {	//get it's coordinates 
*xx=x; *yy=y; }

Bonus::Bonus() {			//constructor
 x=BONUSX; y=BONUSY; 			//just set the coordinates
}

Bonus::~Bonus() {			//destructor
delete g;
}

int Bonus::code() {			//code to be done while it exists
return lifetime--;			//just countdown the lifetime
}

void Bonus::start() {			//reset the bonus
 lifetime=BONUSTIME; 			//set lifetime to a certain amount
}


