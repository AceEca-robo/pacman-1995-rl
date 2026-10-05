#include"gboard.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include"pac.h"

G_Board* G_Board::_instance=0;

G_Blank* G_Board::blank=0;		//initializers
VerticalWall* G_Board::vertical=0;
HorizontalWall* G_Board::horizontal=0;
Cross* G_Board::cross=0;
 G_SpecialWall*  G_Board::specwall=0;
 G_Food*  G_Board::food=0;
 G_SuperFood*  G_Board::superfood=0;
 E0*  G_Board::e0=0;
 E90*  G_Board::e90=0;
 E180*  G_Board::e180=0;
 E270*  G_Board::e270=0;
 T0*  G_Board::t0=0;
 T90*  G_Board::t90=0;
 T180*  G_Board::t180=0;
 T270*  G_Board::t270=0;
 Corner1*  G_Board::corner1=0;
 Corner2*  G_Board::corner2=0;
 Corner3*  G_Board::corner3=0;
 Corner4*  G_Board::corner4=0;

G_Board* G_Board::instance() {//returns the pointer to the instance
 if (_instance==0) 
  _instance=new G_Board;
 return _instance;
}

int G_Board::eqpat(char pat[3][3],char en[10]){	//equality with wildcard
int i;
 for(i=0;i<9;i++) 
  if (en[i]!='?')
   if ((*pat)[i]!=en[i]) return 0; 
return 1;
}
 
 //in:board with coordinates, out: the belonging code for the graphic element
char G_Board::pattern(char brett1[BOARDHEIGHT+1][BOARDWIDTH+1],int y,int x) {
char t,u,  	//some temps
 pat[3][3];	//to contain the pattern to be compared
int i,j;   	//plain counters

switch (t=brett1[y][x]) {	//get the code for the Element

 //if it is simple enough, it don't need translation, just return it
 case '_': case '.': case 'o': case' ': return t; break;}

for(i=-1;i<=1;i++) 	//traverse the elements around x,y
 for(j=-1;j<=1;j++) {
  t='\0';//null
  if (i+x==-1 || i+x==BOARDWIDTH) t=' ';	//if x,y is on border
  if (j+y==-1 || j+y==BOARDHEIGHT) t=' ';	//then space is what's beyond it
  u=brett1[y+j][x+i];				//what's at x+i,y+j coordinates
  if (u=='_') u='O';				//if u is a specialwall then 
						//it is just a wall 

  switch (u) {					//if it's a food or superfood 
						//(but not at x,y) it 
						//counts as space
  case '.': case 'o': u=' '; break;
 } 
 if (t=='\0') pat[j+1][i+1]=u; else pat[j+1][i+1]=t;
  //if if not beyond border then it is u else it is t, space

}

 if (eqpat(pat,"? ?OOO? ?")) return '-';//check if patterns matches and return
 if (eqpat(pat,"?O? O ?O?")) return '|';
 if (eqpat(pat,"  ? OO?O ")) return 'n';
 if (eqpat(pat,"?  OO  O?")) return 'b';
 if (eqpat(pat," O?OO ?  ")) return 'v';
 if (eqpat(pat,"?O  OO  ?")) return 'c';
 if (eqpat(pat,"    O ?O?")) return 'e';
 if (eqpat(pat,"  ? OO  ?")) return 's';
 if (eqpat(pat,"?  OO ?  ")) return 'd';
 if (eqpat(pat,"?O? O    ")) return 'x';
 if (eqpat(pat,"? ?OOO O ")) return 't';
 if (eqpat(pat,"?O  OO?O ")) return 'y';
 if (eqpat(pat," O OOO? ?")) return 'u';
 if (eqpat(pat," O?OO  O?")) return 'i';
 if (eqpat(pat," O OOO O ")) return '+';
 pacwarning("pattern");			//give warning if not returned yet
 pacwarning(*pat);
 return '|';				//just to return something
}

 G_Board::G_Board() {			//constructor 

 if (!UserInterface::instance()) { pacexit("no userinterface"); }

 cross=new Cross;			//make a lot of new graphical elements
 vertical=new VerticalWall;
 horizontal=new HorizontalWall;
 blank=new G_Blank;
 food=new G_Food;
 superfood=new G_SuperFood;
 corner1=new Corner1;
 corner2=new Corner2;
 corner3=new Corner3;
 corner4=new Corner4;
 specwall=new G_SpecialWall;
 e0=new E0;
 e90=new E90;
 e180=new E180;
 e270=new E270;
 t0=new T0;
 t90=new T90;
 t180=new T180;
 t270=new T270;
}

 //translate element on board to graphic element depending on context
 GraphElement *G_Board::graphele(char brett[BOARDHEIGHT+1][BOARDWIDTH+1],int i,int j) {
char t;   
 t=pattern(brett,j,i);
 return chartograph(t);
}

 //translate element on board to graphic element depending on context
 GraphElement *G_Board::graphele(char c) {
return chartograph(c);
}


G_Board::~G_Board() {		//destructor
 delete blank;
 delete cross;
 delete horizontal;
 delete vertical;
 delete food;
 delete superfood;
 delete specwall;
 delete corner1;
 delete corner2;
 delete corner3;
 delete corner4;
 delete e0;
 delete e90;
 delete e180;
 delete e270;
 delete t0;
 delete t90;
 delete t180;
 delete t270;
}

GraphElement *G_Board::chartograph(char c) {	//return translated code
GraphElement *g;
  switch (c) {
  case ' ':g=blank; break;
  case '+':g=cross; break;
  case '|':g=vertical; break;
  case '-':g=horizontal; break;
  case '.':g=food; break;
  case 'o':g=superfood; break;
  case '_':g=specwall; break;
  case 'e':g=e90; break;
  case 's':g=e180; break;
  case 'd':g=e0; break;
  case 'x':g=e270; break;
  case 't':g=t0; break;
  case 'y':g=t90; break;
  case 'u':g=t180; break;
  case 'i':g=t270; break;
  case 'c':g=corner1; break;
  case 'v':g=corner2; break;
  case 'b':g=corner3; break;
  case 'n':g=corner4; break;
  default: { g=blank; /*printf("error%d\n",c);*/} break;
}
return g;
}


 
