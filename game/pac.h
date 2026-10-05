#ifndef __pac_h_
#define __pac_h_

void pacexit(char *);	//exit routine
void pacwarning(char *);//warning routine

void writetext(); //just draw text

//number of ghosts and initial coodinates for all
#define GHOSTS 4
#define GHOST1X 15
#define GHOST1Y 9
#define GHOST2X 17
#define GHOST2Y 9
#define GHOST3X 15
#define GHOST3Y 11
#define GHOST4X 17
#define GHOST4Y 11
#define PACX 16
#define PACY 17
#define BONUSX 16
#define BONUSY 15
 
#define LEVELS 16
                  
#define SUPERTIME 50	//the time pacman is super after eating superfood
#define DEADTIME 25	//the time a ghost stays dead
#define BONUSTIME 50	//the time the bonus is available

#define FOODSCORE 10	//the score you get for eating food
#define GHOSTSCORE 100	//the score you get for the first ghost you eat
#define BONUSSCORE 1000	//the score you get when you eat each bonuspoint
                  
//change my name and you will be sued!!!!!!!!!!!!!!!!!!!!!!!!!!
#define PACTITLE "Pacman v. 1.0 (1995.07.18) \xa9 1995 by Roar Thron\xe6s"
// Copyright 1995 by Roar Thronaes
                  
#endif  


