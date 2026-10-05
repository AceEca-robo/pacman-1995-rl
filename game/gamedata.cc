#include <stdlib.h>
#include"gamedata.h"
#include"board.h"
#include"pac.h"

Gamedata* Gamedata::_instance=0;
long  int Gamedata::hiscore=0;
long int Gamedata::score=0;
long int Gamedata::bonus=0;
int Gamedata::level=0;
int Gamedata::boardlevel=0;
int Gamedata::lives=0;
int Gamedata::eats=0;

Gamedata::Gamedata() { setlevel(0); lives=0; hiscore=0; score=0; bonus=0; eats=foods;}

Gamedata* Gamedata::instance() {      //returns the pointer to the instance
 if (_instance==0) 
  _instance=new Gamedata;
 return _instance;
}

//initializing of gamedata before game starts 
void Gamedata::start(void) { level=1; setboardlevel(); lives=3; score=0; bonus=5000; eats=foods;}

//initializing of gamedata after completed level
void Gamedata::start2(void) { levelup(); 
bonus=5000; eats=foods;}

void Gamedata::scorepluss(int sc) {score+=sc; if (score>hiscore) hiscore=score;}

void Gamedata::bondwn() {if (bonus) bonus-=5;}

void Gamedata::levelup() {level++;
setboardlevel();
}

void Gamedata::livesdwn() {if (lives) lives--;}

void Gamedata::livesup() {lives++;}

long int Gamedata::getscore() {return score;}

long int Gamedata::gethiscore() {return hiscore;}

long int Gamedata::getbonus() {return bonus;}

int Gamedata::getlives() {return lives;}

int Gamedata::getlevel() {return level;}

void Gamedata::resetscore() {score=0;}

void Gamedata::setbonus(long int b) {bonus=b;}

void Gamedata::setlevel(int l) {level=l; setboardlevel();} 

void Gamedata::setboardlevel() {
 if (level>LEVELS) boardlevel=1+(random()%LEVELS);
 else boardlevel=level;
}
 
int Gamedata::getboardlevel() {
return boardlevel;
}

void Gamedata::setlives(int l) {lives=l;}

void Gamedata::reseteats() {eats=foods; }

int Gamedata::eatenall() {return !eats; }

void Gamedata::foodeat() {if (eats) eats--; else pacexit("food"); }



