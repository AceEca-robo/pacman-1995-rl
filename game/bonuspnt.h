#ifndef __bonuspnt_h_ 
 #define __bonuspnt_h_
 #include"bonus.h"

class BonusPoint : public Bonus {
   
public:

BonusPoint();			//constructor
typ is_a(void) {return classBonusPoint;}    
void eat(Gamedata*);
  
};

#endif


