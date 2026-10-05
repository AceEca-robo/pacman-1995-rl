#ifndef __gamedata_h_
#define __gamedata_h_
#include"object.h"

class Gamedata : public Object {

static Gamedata* _instance;
static long  int hiscore,score,bonus;
static int level,boardlevel,lives,eats;

/* static */ void setboardlevel();

 protected:
Gamedata();

 public:
static Gamedata* instance();
/* static */ void start(void);	//initializing of gamedata before game starts 
/* static */ void start2(void);	//initializing of gamedata after completed level
/* static */ void scorepluss(int);	//the rest if simple
/* static */ void bondwn();
/* static */ void levelup();
/* static */ void livesdwn();
/* static */ void livesup();
/* static */ void foodeat();
/* static */ long int getscore();
/* static */ long int gethiscore();
/* static */ long int getbonus(); 
/* static */ int getlevel();
/* static */ int getboardlevel();
/* static */ int getlives();
/* static */ int eatenall();
/* static */ void resetscore();
/* static */ void reseteats();
/* static */ void setbonus(long int); 
/* static */ void setlevel(int);
/* static */ void setlives(int);
};

#endif
