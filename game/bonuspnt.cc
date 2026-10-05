#include"bonuspnt.h"
#include"gbonpnt.h"
#include"pac.h"

BonusPoint::BonusPoint():Bonus() { 
 g=new G_BonusPoint; 		//new one
 }

void BonusPoint::eat(Gamedata* da) { 
 da->scorepluss(BONUSSCORE);	//specific action: increase score 
}

