#include"pacman.h"
#include"types.h"
#include"pac.h"

Pacman::Pacman() {	//constructon
 b=Board::instance();	//get the instance of the board
 g=new G_Pacman;	//a new one
 supertime=0;		//supertime must be set to zero
}

Pacman::~Pacman() {	//destructor
delete g;
}

void Pacman::start() {
 x=PACX; 		//set it's initial coordinates
 y=PACY;
 st=normal;		//set status to normal
 d=still; 		//direction is still
 try_d=right;		//face right
 supertime=0;		//supertime must be set to zero
}

int Pacman::is_dead() {	//is pacman dead?
 return st==dead;
}

void Pacman::die(Gamedata *da) {	//let pacman die a life
 st=dead; 				//status is dead
 da->livesdwn();			//decrement a life
}

int Pacman::getsupertime() {	//get the remaning time pacman is super
return supertime;}

int Pacman::is_super() { 	//is pacman super?
return st==super; }

void Pacman::draw(void) {	//specific draw for pacman
 g->draw(x,y,d,try_d);	//draw pacman direction it was going and was tried
}

GID_TYPE Pacman::getgid() {
return g->getgid(d,try_d);
}

void Pacman::getxy(int *xx,int *yy) {
*xx=x;
*yy=y;
}

int Pacman::go(direction dd,Gamedata *da) {//what direction to try to move it
typ w;					//what's at new coordinates
int i=1;	//this i is returned: it is whether pacman moved or not.
int xx,yy;	//possible next coordinates
if (dd!=none) try_d=d=dd;//if a direction!=none given, set the two directions
if (st==dead) { 	//if pacman died
d=still; 		//then actual moving direction is still 
return 0;		//and return not moved
}
switch (st) {
 case super: {		//if super
  if (supertime) 	//if supertime is > 0
   supertime--; 	//decrement supertime
  else 
   st=normal;   	//if supertime is zero then status is normal
} break;
 case normal: break;
}

next(&xx,&yy,d,x,y);	//compute next coordinates
w=b->what_is(xx,yy);	//what's at new coordinates

switch (w) {		//if it's
 case classSpecialWall: case classWall: //can't move through walls
  { i=0; d=still; } break;
 case classFood: { 
  x=xx; y=yy; 				//set to new coordinates
  b->eat(x,y);				//eat food at board coordinates
  da->foodeat(); 			//let gamedata know a food is eaten
  da->scorepluss(FOODSCORE); 		//increase the score
 } break;
 case classSuperFood: { 
  x=xx; y=yy; 				//set to new coordinates
  b->eat(x,y); 				//eat superfood at board coordinates
  st=super; 				//set status to super
  supertime=SUPERTIME;			//set supertime, too 
 } break; 
 case classBlank: { 
  x=xx; y=yy;  				//set to new coordinates
 } break;
// case BonusLife: { x=xx; y=yy; b->eat(x,y); } break;
// case BonusPoint: { x=xx; y=yy; b->eat(x,y); } break;
 default: /*printf("caseerror\n");*/ break;
}	
//printf("%d\n",d);	   
//printf("%s",w);	   
return i;	//this i is returned: it is whether pacman moved or not.
}


