#ifndef __bonuslif_h_ 
 #define __bonuslif_h_
 #include"bonus.h"

class BonusLife : public Bonus {
   
public:

BonusLife();			//constructor
typ is_a(void) {return classBonusLife;}    
void eat(Gamedata*);
  
};

#endif
