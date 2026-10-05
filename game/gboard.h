#ifndef __gboard_h_ 
#define __gboard_h_
 #include"object.h"
 #include"graphele.h"
#include"gblank.h"
#include"vertwall.h"
#include"horiwall.h"
#include"cross.h"
#include"gspecwal.h"
#include"gfood.h"
#include"gsupfood.h"
#include"e0.h"
#include"e90.h"
#include"e180.h"
#include"e270.h"
#include"t0.h"
#include"t90.h"
#include"t180.h"
#include"t270.h"
#include"corner1.h"
#include"corner2.h"
#include"corner3.h"
#include"corner4.h"
#include"sizes.h"
#include"usrinter.h"
#include"types.h"

#include"element.h"
#include"dynamiel.h"

class G_Board : public Object {

//pointers to the graphic elements

static G_Blank* blank;
static VerticalWall* vertical;
static HorizontalWall* horizontal;
static Cross* cross;
static G_SpecialWall* specwall;
static G_Food* food;
static G_SuperFood* superfood;
static E0* e0;
static E90* e90;
static E180* e180;
static E270* e270;
static T0* t0;
static T90* t90;
static T180* t180;
static T270* t270;
static Corner1* corner1;
static Corner2* corner2;
static Corner3* corner3;
static Corner4* corner4;

int eqpat(char [3][3],char[10]);	//equality with wildcard

char pattern(char [BOARDHEIGHT+1][BOARDWIDTH+1],int,int);
 //in:board with coordinates, out: the belonging code for the graphic element

GraphElement *chartograph(char);
 //"translates" graphic element code to pointer to it

static G_Board* _instance;	 //pointer to the instance of the board
 protected:

 G_Board();             	//constructor
 
 public:

 static G_Board* instance();	//returns the pointer to the instance 
 ~G_Board();			//destructor

 GraphElement *graphele(char [BOARDHEIGHT+1][BOARDWIDTH+1],int,int);
  //translates board info to pointer to graphic element

 GraphElement *graphele(char);
  //the same
};

#endif  






