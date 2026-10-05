#include"bonuslif.h"
#include"gbonlife.h"
#include"pac.h"

BonusLife::BonusLife():Bonus() { 
 g=new G_BonusLife; 	//new one
 }

void BonusLife::eat(Gamedata* da) { 
 da->livesup(); 	//specific action: increment lives
}
