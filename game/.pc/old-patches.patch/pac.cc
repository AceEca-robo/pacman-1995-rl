#include<stdio.h>    
#include<stdlib.h>
#include"pac.h"
#include"gamedata.h"
#include"usrinter.h"
#include"pacman.h"
#include"board.h"
#include"direc.h"
#include"ghost.h"
#include"bonuslif.h"
#include"bonuspnt.h"
#include"timing.h"
#include"arg.h"
 
// routine to make the game exit if there was a previous detected error
 
void pacexit(char *s) { 
#ifdef XWIN
printf("%s\n",s); exit(1); 
#endif
}

// routine do display a warning if it should be

void pacwarning(char *s) { 
#ifdef XWIN 
printf("Warning: %s\n",s);
#endif
}

// checks whether pacman has eaten the bonus, or pacman or some ghost
// have met and one of them must "die"
 
void check(Board* bo,Gamedata *da,Pacman* pa,Ghost *gh[GHOSTS],Bonus *&bon) {
 int i;				//counter
 int alive;			//is pacman still alive 
 int x1, y1;			//pacman coordinates
 int x2, y2;			//coordinates for bonus or a ghost
 pa->getxy(&x1,&y1);		//get pacman coordinates
 if (bon) {			//if there is a bonus on the board
  bon->getxy(&x2,&y2);		//get bonus coordinates
  if (x1==x2 && y1==y2) {	//if pacman and bonus coordinates the same
   bon->eat(da);		//then bonus is eaten
   bon=0;} 			//and gone
 }
 for(i=0,alive=1;alive && i<GHOSTS;i++) {	//for all ghosts while p. alive	
  gh[i]->getxy(&x2,&y2);	//get coordinates
  if (x1==x2 && y1==y2) {	//if pacman and ghost coordinates the same
   if (pa->is_super()) 		//if pacman is super
    { 
     gh[i]->die(da);		//then ghost dies	
    } else { 
     pa->die(da);		//else pacman dies
     alive=0;			//set not alive
   }
  } 
 }
}

void writetext() { //just draw text

UserInterface *u=UserInterface::instance();
Gamedata *da=Gamedata::instance();

u->write(1,1,"Hiscore");        //print to userinterface at x, y, something
u->write(1,2,da->gethiscore(),8);
u->write(1,4,"Score");
u->write(1,5,da->getscore(),8);
u->write(1,7,"Lives");
u->write(1,8,da->getlives(),8);
u->write(1,10,"Level");
u->write(1,11,da->getlevel(),8);
u->write(1,13,"Bonus");
u->write(1,14,da->getbonus(),8);
u->write(1,16,"Supertime");
}

#ifdef MSWIN
int PASCAL WinMain(HANDLE hInstance, HANDLE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
UserInterface::sethInstance(hInstance);	//storing info from MS Window Manager
UserInterface::sethPrevInstance(hPrevInstance);
UserInterface::setlpCmdLine(lpCmdLine);
UserInterface::setnCmdShow(nCmdShow);
#else
main(int argc,char **argv) {
Argument(argc, argv); 		// storing the command line arguments
#endif

int i,j; 			//plain counters
Gamedata *da=Gamedata::instance(); //gamedata like score, hiscore, lives, levels etc
Bonus *bon=0; 			//the bonus, it can be bonuslife or bonuspoint
int g; 				//whether pacman moved
int inp; 		//the result of getinput: whether the game is done
int done; 			//whether game is done
direction d; 			//the direction pacman moved
UserInterface* u=UserInterface::instance(Keyboard); 	//the userinterface to the user

srandom(rand());		//reset random number generator

Board* board=Board::instance(); //the pacman board matrix
Pacman pac; 			//the pacman itself
Ghost *gh[GHOSTS]; 		//the ghosts
BonusLife *bonl=new BonusLife;	//a bonuslife not connected to bonus yet
BonusPoint *bonp=new BonusPoint;//a bonuspoint not connected to bonus yet
gh[0]= new Ghost(Colour::GHOSTCOLOUR1,GHOST1X,GHOST1Y); //initializing ghost 1 
gh[1]= new Ghost(Colour::GHOSTCOLOUR2,GHOST2X,GHOST2Y); //initializing ghost 2
gh[2]= new Ghost(Colour::GHOSTCOLOUR3,GHOST3X,GHOST3Y); //initializing ghost 3
gh[3]= new Ghost(Colour::GHOSTCOLOUR4,GHOST4X,GHOST4Y); //initializing ghost 4
    
inp=u->getinput(); 		//gets whether the game is done

writetext();

done=0; 			//means not done
while (!done) { 		//while not done

 da->start(); 			//total initialization of the gamedata
u->write(1,8,da->getlives(),8);
u->write(1,11,da->getlevel(),8);
u->write(1,14,da->getbonus(),8);

while (!done && da->getlives()) { //while game not done and pacman still alive

board->start(da->getboardlevel());//resets board to a certain level

pac.start(); 			//reset pacman
for(i=0;i<GHOSTS;i++) 		//reset all ghosts
 gh[i]->start();

board->draw(); 			//draw entire board


d=still; 			//pacman got direction still, no moving
board->delta_draw(); 		//draw just the changes of the board
timingstart(); 			//reset the timer

while (done==0 && da->getlives() && !da->eatenall()) { 
				//while game not done and pacman still alive 
				// and while not all food eaten

 u->write(1,8,da->getlives(),8);
 for(i=0;i<GHOSTS;i++) 		//let all ghosts look for pacman
  gh[i]->lookforpac(&pac);
 g=pac.go(d,da); 		//let pacman go direction d
 for(i=0;i<GHOSTS;i++)
  gh[i]->go(&pac); 		//let the ghosts move
 check(board,da,&pac,gh,bon);	// checks whether pacman has eaten the bonus, or pacman or some ghost have met and one of them must "die" 
 board->sprite(&pac); 		//let it be a sprite
 for(i=0;i<GHOSTS;i++)
  board->sprite(gh[i]);		//let it be a sprite
 if (bon) board->sprite(bon);	//let it be a sprite
 board->delta_draw(); 		//draw just the changes of the board
 timing(pac.is_super()); 	//do timing: make the game go smooth enough
 d=u->stick();			//pacman got direction
 if (!bon) { 			// if no bonus if "on"
  if ((random() & 4095)==0) { 
   if (random() & 61440) 
    bon=bonp; 			//if seldom start a bonuspoint
     else 
    bon=bonl;		 	//if very seldom start a bonuslife
   bon->start(); } 		//reset the spesific bonus thing
  } 
 if (bon) { 			//if bonus exists
  if (!bon->code()) 		//returns whether the bonus has lived it's time
   bon=0; } 			// "remove" bonus
 if (pac.is_super()) { 		//if pac has eaten superfood do an extra run
  u->write(1,17,pac.getsupertime(),8);
  g=pac.go(d,da);		//let pacman go direction d
  check(board,da,&pac,gh,bon);	// checks whether pacman has eaten the bonus, or pacman or some ghost have met and one of them must "die" 
  board->sprite(&pac);		//let it be a sprite
  for(i=0;i<GHOSTS;i++)  
   board->sprite(gh[i]);	//let it be a sprite
  if (bon) board->sprite(bon);	//let it be a sprite
  board->delta_draw();		//draw just the changes of the board
  timing(pac.is_super());	//do timing: make the game go smooth enough
  d=u->stick();			//pacman got direction
 }
 if (pac.is_dead()) {  		// do this if pacman died
      bon=0;                    //no more bonus, is gone
      u->write(1,8,da->getlives(),8);
      board->sprite(&pac);	//let it be a sprite
      board->delta_draw();	//draw just the changes of the board
      timing(0);		//do timing
      pac.go(right,da);	//let pacman point right
      board->sprite(&pac);	//let it be a sprite
      board->delta_draw();	//draw just the changes of the board
      timing(0);		//do timing
      pac.go(up,da);		//let pacman point up
      board->sprite(&pac);	//let it be a sprite
      board->delta_draw();	//draw just the changes of the board
      timing(0);		//do timing
      pac.go(left,da);		//let pacman point left
      board->sprite(&pac);	//let it be a sprite
      board->delta_draw();	//draw just the changes of the board
      timing(0);		//do timing
      pac.go(down,da);		//let pacman point down
      board->sprite(&pac);	//let it be a sprite
      board->delta_draw();	//draw just the changes of the board
      timing(0);		//do timing 
      pac.go(right,da);	//let pacman point right
      board->sprite(&pac);	//let it be a sprite
      board->delta_draw();	//draw just the changes of the board
      timing(0);		//do timing
      pac.start();		//reset pacman
      for(i=0;i<GHOSTS;i++) 
       gh[i]->start(); 		//reset ghosts
 }

 da->bondwn();			//countdown bonus
 u->write(1,2,da->gethiscore(),8);
 u->write(1,5,da->getscore(),8);
 u->write(1,14,da->getbonus(),8);
 if (pac.is_super()) 		//if pac is super
  u->write(1,17,pac.getsupertime(),8);	//write remaining supertime
 else 
  u->write(1,17,"         ");	//write nothing

  inp=u->getinput();		//get whether the game is wanted to end
  done=inp;
  }

if (!done) 			//if not done
if (!da->eatenall()) { 		//if not all food eaten
 u->write(1,20,"GAME OVER"); 	//then the game must be over
 u->waitsync(); 		//wait til everything is printed to screen
 for(i=0;i<4;i++,timing(0)); 	//wait a sec or something
 u->write(1,20,"         "); 	//then write nothing
 da->start(); 			//reset gamedata
 u->write(1,8,da->getlives(),8);
 u->write(1,11,da->getlevel(),8);
 u->write(1,14,da->getbonus(),8);
} else {  			//all food must have been eaten
 da->scorepluss(da->getbonus()); 	//add score with bonus
 da->start2(); 			//prepare gamedata for next level
 u->write(1,11,da->getlevel(),8);
 u->write(1,14,da->getbonus(),8);
}
}
}

// all the rest of this is end of game cleanup

delete bonp;
delete bonl;
for(i=0;i<GHOSTS;i++) delete gh[i];
      
return 1; 
          
}
  
                
           
