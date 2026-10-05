#include"board.h"
#include"pac.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>


Board* Board::_instance=0; 	//initializers

liststruct* Board::list=0;
liststruct* Board::oldlist=0;
Element* Board::board[BOARDWIDTH][BOARDHEIGHT]=initen;

G_Board* Board::gboard=0;

UserInterface* Board::u=0;

Board* Board::instance() {	//returns the pointer to the instance
 if (_instance==0) 
  _instance=new Board;
 return _instance;
}

 Board::Board() {		//constructor 

gboard=G_Board::instance();	//get instances of what we need
u=UserInterface::instance();

 if (!u) { pacexit("no userinterface"); }

 list=0; 			//initializers to null

int i,j;
for(j=0;j<BOARDHEIGHT;j++)
 for(i=0;i<BOARDWIDTH;i++) 
  board[i][j]=0;       

}

void Board::start(int level) {	//initialize board for a given level

int i,j; 			//counters
Element *g;			//pointer to the presently traversed element
char t;    			//what code for Element is it
for(j=0;j<BOARDHEIGHT;j++) 	//traverse all
 for(i=0;i<BOARDWIDTH;i++) {        
  switch (t=boards[level-1][j][i]) {	//cases for the code
  case ' ':g=new Blank; break;
  case 'O':g=new Walls; break;
  case '.':g=new Food; break;
  case 'o':g=new SuperFood; break;
  case '_':g=new SpecialWall; break;
  default: { g=new Blank; pacwarning("unrecognized");} break;
}
//get pattern translated to graphic element
g->gset(gboard->graphele(boards[level-1],i,j));
if (board[i][j]) delete board[i][j];
board[i][j]=g;
}                                     
}

void Board::replace(Element*e) {}
int i,j;

Board::~Board() {			//destructor
delete gboard;

for(j=0;j<BOARDHEIGHT;j++)
 for(i=0;i<BOARDWIDTH;i++)
  delete board[i][j]; 
}

void Board::draw(void) {		//draw entire board

int i,j;
for(j=0;j<BOARDHEIGHT;j++)
 for(i=0;i<BOARDWIDTH;i++)
  board[i][j]->draw(i,j);
list=oldlist=0;			//reset spritelists
u->waitsync();			//don't return until everything is drawn
} 
                         
 void Board::delta_draw(void) {		//draw changes of the board
 int xx,yy; 				//coordinates
 int zero;
 liststruct *next, *oldnext, *temp, *oldtemp;	//pointers to liststruct/element, current and to be displayed, and belonging temporaries

 oldtemp=oldlist;
 while (oldtemp) {		//traverse the list of last sprites displayed
  xx=oldtemp->x;		//get coordinates of current element
  yy=oldtemp->y;
  temp=list;
  // traverse the list of sprites to be display and check coordinates
  while (temp && (xx!=temp->x || yy!=temp->y)) temp=temp->next;
  zero=(xx==0 && yy==0); 		//if coordinates are both zero
  if (!temp)  board[xx][yy]->draw(xx,yy); //if temp==0, that is, if there is no
                                          //new sprite to be displayed where
                                          //a sprite now is display, then
                                          //redraw the element from the board
  oldtemp=oldtemp->next;
 }

 temp=list;
 while (temp) {			//traverse the list of sprites to be displayed
  xx=temp->x;			//get coordinates of current element
  yy=temp->y;
  oldtemp=oldlist;
  // traverse the list of sprites currently display and check coordinates and if equal if the current and new grafical ids are the equal
  while (oldtemp && (xx!=oldtemp->x || yy!=oldtemp->y || temp->gid!=oldtemp->gid)) 
   oldtemp=oldtemp->next;
//if (oldtemp) printf("xxx%d %d %d\n",oldtemp->g,oldtemp->x,oldtemp->y);
  if (!oldtemp) temp->g->draw();//if old temp==0, that 
                                  //is, if there is no identical
                                          //new sprite to be displayed where
                                          //a sprite now is display, then
                                          //display the new sprite
  temp=temp->next;
 }
 oldtemp=oldlist;
 while (oldtemp) {	//delete elements in the now previous sprite list
  oldnext=oldtemp->next;
  delete oldnext;
  oldtemp=oldnext;
 }
if (zero && oldlist) {	//personal thingie used for debug, not useful
 oldtemp=oldlist;
 while (oldtemp) {
  pacwarning("zero");
 // printf("%d %d %d   ",oldtemp->gid,oldtemp->x,oldtemp->y);
  oldtemp=oldtemp->next;
 } // printf("\n");
 temp=list;
 while (temp) {
  pacwarning("zero");
//  printf("%d %d %d   ",temp->gid,temp->x,temp->y);
  temp=temp->next;
 } // printf("\n\n");
}
 oldlist=list;
 list=0;
//printf("here4\n");

u->waitsync();		//don't return until everything is drawn
}

 typ Board::what_is(int xx,int yy) {//what "class" element at coordinates is 
return board[xx][yy]->is_a();	//return what kind of element is at xx,yy
}  

 void Board::eat(int xx,int yy) {	//removes whats at coordinates
if (board[xx][yy]) delete board[xx][yy];//delete the element
board[xx][yy]=new Blank;//get a new one
board[xx][yy]->gset(gboard->graphele(' '));//set it to space=blank

}

void Board::sprite(DynamicElement *g) {	//let dynamicelement be a sprite
 liststruct *temp=0,*last=0;		//resets to null
 liststruct *ny=new liststruct;		//get a new liststruct
 ny->g=g;				//get pointer to element
 ny->gid=g->getgid();			//get it's grafical id
 g->getxy(&ny->x,&ny->y);		//and coordinates

 temp=list;
 while (temp && (temp->x!=ny->x || temp->y!=ny->y)) {//try to find next last 
  last=temp; 
  temp=temp->next; 
 }
 //insert at last place
 if (!temp) {				//if the list was null it must be first
  ny->next=temp;
  if (temp==list) 
   list=ny;
  else 					//put at back
   last->next=ny;
 }
/*
 ny->next=list;
 list=ny;
*/
/*
 temp=list;
 while (temp) {
  printf("%d %d\n",temp,temp->g);
//  temp->g->draw();
printf("%d %d\n",temp->x,temp->y);
printf("hehere\n");
  temp=temp->next;
 }
printf("\n\n");
*/

}

 
