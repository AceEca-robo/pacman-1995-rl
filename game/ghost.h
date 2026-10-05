#ifndef __ghost_h_ 
#define __ghost_h_
#include"moveable.h"
#include"board.h"
#include"usrinter.h"
#include"gghost.h"
#include"direc.h"
class Pacman;
#include"pacman.h"

typedef enum {randm,hunter,hunted,eaten} gstat;
//the status: whether the ghost is going at random, it has seen pacman and is hunting him down, if pacman is super and hunting the ghosts, and whether the ghost itself is eaten by the pacman

class Ghost : public Moveables {
int startx,starty, 	//starting coordinates
    go_x,go_y, 		//if it hunts pacman, this is were it was last seen
    dx,dy;     		//increments to get go where pacman was last seen     
gstat st;		//the ghost's status
direction d;		//the direction the ghost is going
int deadtime;	//the remaing time for the ghost to be dead if it's eaten
int randmv;	//a random number for changing direction when moving at random
G_Ghost *g;		//a pointer to the ghost graphical element
static int ghostseaten;	//the number of ghosts eaten by pacman while super

void newdir(int,int,direction*);	//gets a new direction to go
int otherdir(int,int,direction); 	//is there an other direction to go
int searchpac(direction *,int *,int *,Pacman *);	//look for pacman

public:

Ghost(COLOURTYPE,int,int);		//constructor with coordinates
~Ghost();				//destructor
void draw(void);			//draw the ghost
int go(Pacman*);				//go for pacman: do ghost code
typ is_a(void) {return classGhost;}    
void eat(void);				//when pacman eats ghost
void getxy(int*,int*);
void die(Gamedata*);			//when pacman eats ghost
void start();				//reset ghost
GID_TYPE getgid();
void lookforpac(Pacman *pac);		//look for pacman
};

#endif
