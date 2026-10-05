#ifndef __pacman_h_ 
#define __pacman_h_
#include"moveable.h"
#include"board.h"
#include"usrinter.h"
#include"gpacman.h"
#include"direc.h"

typedef enum {dead,normal,super} stat;

class Pacman : public Moveables {
G_Pacman *g;		//pointer to the pacman graphic element
stat st;		//what't it's state: dead, normal or super
int supertime;		//the remaning time pacman is super
direction d,try_d;	//what direction is it going, what direction was tried

public:

int getsupertime();		//get the remaning time pacman is super
void start(); 			//reset pacman
int is_dead(); 			//is pacman dead?
void die(Gamedata*);		//let pacman die a life
Pacman();			//constructon
~Pacman();			//destructor
void draw(void);		//specific draw for pacman
int go(direction,Gamedata*);	//what direction to try to move pacman
void getxy(int*,int*);
typ is_a(void) {return classPacman;}    
int is_super(void); 		//is pacman super?
GID_TYPE getgid();  
};

#endif


