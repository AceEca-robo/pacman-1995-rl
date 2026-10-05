#include"ghost.h"

#include<stdlib.h>
#include"types.h"
#include"pac.h"

int Ghost::ghostseaten=0;			//initializing

Ghost::Ghost(COLOURTYPE farge,int xx,int yy) {	//constructor
 b=Board::instance();				//get the instance of the board
 d=up; 						//direction to move is up
 g=new G_Ghost(farge);				//a new one
 startx=x=xx;					//set initial coordinates
 starty=y=yy;
 st=randm;					//move at random
 randmv=4;
 randmv=random()%8 +4;//a random number about changing direction when randoming
}

Ghost::~Ghost() {
delete g;
}

void Ghost::start() {	//reset ghost
 x=startx;		//set initial coordinates
 y=starty;
 st=randm;		//move at random
 d=up;			//direction is up
 ghostseaten=0;		//number of ghosts eaten is zero
}

GID_TYPE Ghost::getgid() {
return g->getgid();
 }
 
void Ghost::die(Gamedata* da) {	//when pacman eats ghost
int i, 				//plain counter
   sc; 				//score temporary
 ghostseaten++;			//another ghost eaten by pacman
 for(i=ghostseaten,sc=GHOSTSCORE;i>1;i--,sc*=2);
 da->scorepluss(sc);
 deadtime=DEADTIME;		//set time to be dead
 st=eaten;			//set status to eaten
 x=startx;			//set initial coordinates
 y=starty;
 d=still;			//direction is still
}

void Ghost::getxy(int *xx,int *yy) { *xx=x; *yy=y; }

void Ghost::draw(void) {	//draw the ghost
 g->draw(x,y,st==hunted);//3rd parameter is about colour: ghost hunted or not
}

void Ghost::eat(void) {		//when pacman eats ghost
 deadtime=DEADTIME;		//set time to be dead
 st=eaten;			//set status to eaten
 x=startx;			//set initial coordinates
 y=starty;
}

int Ghost::otherdir(int xx,int yy,direction dd) {//is there an other direction to go
 int xxx,yyy;			//coordinates
 direction goleft,goright;	//the relative left/right to where a ghost goes
 switch (dd) {			//find out relative direction
 case up: {goleft=left; goright=right; } break;
 case down: {goleft=right; goright=left; } break;
 case left: {goleft=up; goright=down; } break;
 case right: {goleft=down; goright=up; } break;
	}
 next(&xxx,&yyy,goleft,xx,yy);	//next coordinates if left
 if (b->what_is(xxx,yyy)==classWall) goleft=none;//set none if a wall is left
 next(&xxx,&yyy,goright,xx,yy);//next coordinates if right
 if (b->what_is(xxx,yyy)==classWall) goright=none;//set none if a wall is right
// printf("%d\n",goleft!=none && goright!=none);
 return (goleft!=none || goright!=none);//whether left or right is not a wall
}

void Ghost::newdir(int xx,int yy,direction* dd) {//gets a new direction to go
 int xxx,yyy;			//coordinates
 direction goleft,goright;	//the relative left/right to where a ghost goes
 switch (*dd) {			//find out relative direction
 case up: {goleft=left; goright=right; } break;
 case down: {goleft=right; goright=left; } break;
 case left: {goleft=up; goright=down; } break;
 case right: {goleft=down; goright=up; } break;
	}
 next(&xxx,&yyy,goleft,xx,yy);	//next coordinates if left
 if (b->what_is(xxx,yyy)==classWall) goleft=none;//set none if a wall is left
 next(&xxx,&yyy,goright,xx,yy);//next coordinates if right
 if (b->what_is(xxx,yyy)==classWall) goright=none;//set none if a wall is right
//if (goleft==none && goright==none) exit(1);
//if either is none return the other
// if (startx==15 && starty==9) printf("change at %d %d\n",x,y);
 if (goleft==none) { *dd=goright; return; }
 if (goright==none) { *dd=goleft; return; }
//if both is not none then choose one at random
 if (random()%2)*dd=goright; else *dd=goleft;
}

int Ghost::searchpac(direction *dd,int *gox,int *goy,Pacman *pac) {//look for pacman
 int xx,yy,			//coordinates
 dx,dy,				//delta goto coordinates
 goxx,goyy;			//goto coordinates
  pac->getxy(&goxx,&goyy);	//get pacman coordinates
  if (goxx==x || goyy==y) {	//if either x or y coordinates same
   dx=goxx-x;
   dy=goyy-y;
//set deltas dx dy to add to x,y to get to goxx goyy
   if (dx<0) dx=-1;
   if (dx>0) dx=1;
   if (dy<0) dy=-1;
   if (dy>0) dy=1;
   xx=x; yy=y;
   while (!(xx==goxx && yy==goyy) && b->what_is(xx,yy)!=classWall) {
    xx+=dx; yy+=dy;  }
//checks whether there is a wall between pacman and ghost
   if (xx==goxx && yy==goyy) {//if no wall hunt pacman
    *gox=goxx; *goy=goyy;
    if (st==hunted) { dx=-dx; dy=-dy; } //if pacman is super then run
    if (dx==0 && dy==-1) *dd=up;
    if (dx==0 && dy==1) *dd=down;
    if (dx==1 && dy==0) *dd=right;
    if (dx==-1 && dy==0) *dd=left;
    return 1;
   }
 }
 return 0;
}

void Ghost::lookforpac(Pacman *pac) {//look for pacman
int i=1;			//ghost moved?
int xx,yy;			//coordinates

if (pac->is_super())		// if pacman is super 
 { if (st!=eaten) st=hunted; }	//then if ghost is not eaten then it is hunted
else { 				//if pacman not super
 if (st==hunted) {	//if ghost is (was) hunted it must now be random
  st=randm; 			//must be random
  ghostseaten=0;		//set number of ghosts eaten back to zero
 } 
} 
switch (st) {			//case ghost status of
 case hunter: {			//if ghost is hunter
 i=searchpac(&d,&go_x,&go_y,pac);//then search for pacman
if (i) {			//if found 
 pac->getxy(&xx,&yy);		//and pacman coordinates
 if (x==xx && y==yy) d=still;	//are equal to ghosts's: stand still
}
}
  break;
 case randm: {
  i=searchpac(&d,&go_x,&go_y,pac);	//then search for pacman
  if (i) {				//if found
   st=hunter;				//then ghost is a hunter
  } else {
  }
 
} break;
 case hunted: {
  i=searchpac(&d,&go_x,&go_y,pac);	//then search for pacman
} break;
 case eaten: {
} break;
	    }
}

Ghost::go(Pacman *pac) {	//go for pacman: do ghost code
typ w;				//what type is at next coordinates
int i=1;			//moved or not?
int xx,yy;			//coordinates

if (pac->is_super()) 		//if pacman is super
 { if (st!=eaten) st=hunted; } 	//if st is not eaten then ghost is hunted
else { if (st==hunted) st=randm; } 	//else if pacman is not and ghost
                                   	//is still hunted then it is now random
switch (st) {			//if ghost status is
 case hunter: {			//hunter
 i=searchpac(&d,&go_x,&go_y,pac);//then search for pacman
 if (i) { 			//if found
  pac->getxy(&xx,&yy);		//and pacman coordinates
  if (x==xx && y==yy) d=still;	//are equal to ghosts's: stand still
  break;
 }
 if (go_x==x && go_y==y) {//if ghost has reached pacman last known coordinates
  pac->getxy(&xx,&yy);	//get pacman coordinates
//if (!otherdir(x,y,d)) //and if there is not an other possible l/r direction
  newdir(x,y,&d); 	//change direction
  st=randm; 		//and do random movements.
 }
} break;
 case randm: {				//if ghost moves at random
  i=searchpac(&d,&go_x,&go_y,pac);	//then search for pacman
  if (i) {				//if found
   st=hunter;				//then ghost is hunter
  } else {
   if (otherdir(x,y,d)) {//and if there is an other possible l/r direction
    if (!(random()%randmv)) 	//if random "hits" 
     newdir(x,y,&d); 		//then change direction
  }
 }
} break;
 case hunted: {			//if the ghost is hunted
  i=searchpac(&d,&go_x,&go_y,pac);//then search for pacman
} break;
 case eaten: {			//if ghost is eaten
  if (deadtime) 		//is > 0
   deadtime--;  		//decrement it by one
  else 
 {
  st=randm; 		//else, if it is zero, then it's status is random
  d=up; 		//and direction is up
 }
} break;
	    }
//if (dd!=none) d=dd;
next(&xx,&yy,d,x,y);		//find next coordinates
w=b->what_is(xx,yy);		//what's at those new coordinates then
//if (x==xx && y==yy) printf("%d %d\n\n",st,d);
//printf("%d %d %d %d %d %d %d %d %d
//%d\n\n",up,down,left,right,none,still,randm,hunter,hunted,eaten);
//exit(1);
switch (w) {//if w is a
// case SpecialWall: { x=xx; y=yy; } break;
 case classWall: { 
  i=0; 				//can not go through walls
  newdir(x,y,&d); 		//find a new direction
  next(&xx,&yy,d,x,y); 		//find next coordinates
  x=xx; y=yy; 			//set to new coordinates 
 } break;
// case Food: { x=xx; y=yy; } break;
// case SuperFood: { x=xx; y=yy;  } break; 
// case Blank: { x=xx; y=yy; } break;
// case BonusLife: { x=xx; y=yy; } break;
// case BonusPoint: { x=xx; y=yy; } break;
 default: { x=xx; y=yy; } break;//this is the standard: set to new coordinates
// default: printf("caseerror\n"); break;
}	
//printf("%s",w);	   
return i;
}


//   if (otherdir(x,y,d)) {//and if there is an other possible l/r direction
//    if (!(random()%randmv))	//if random "hits" 
//     newdir(x,y,&d); 	//then change direction


// if (go_x==x && go_y==y) {//if ghost has reached pacman last known coordinates
//  pac->getxy(&xx,&yy);	//get pacman coordinates
//if (!otherdir(x,y,d));	//and if there is an other possible l/r direction
//   newdir(x,y,&d);	//change direction
//   st=randm;		//and do random movements.
