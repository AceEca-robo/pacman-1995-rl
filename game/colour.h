#ifndef __colour_h_
#define __colour_h_

#include"systspec.h"
#include"usrinter.h"
#include"object.h"

//defines the type of the colour types
#ifdef XWIN
#define COLOURTYPE unsigned long
#define COLOURNAME XColor
#elif defined MSWIN
#define COLOURTYPE COLORREF
#define COLOURNAME COLORREF
#elif defined VTX
#define COLOURTYPE int
#define COLOURNAME int
#endif

class Colour : public Object {

#ifdef XWIN			//X-Win only
Display* display;		//pointer to display connection
Colormap cmap;			//a colourmap
#endif

 public:
Colour(UserInterface*);		//constructor and initializer
~Colour();			//destructor
                           
static COLOURTYPE MYFOREGROUND; //foreground colour
static COLOURTYPE MYBACKGROUND; //background colour
static COLOURTYPE WALLCOLOUR;   //colour for wall
static COLOURTYPE PACMANCOLOUR; //colour for pacman
static COLOURTYPE GHOSTCOLOUR1; //colour for ghost 1
static COLOURTYPE GHOSTCOLOUR2; //colour for ghost 2
static COLOURTYPE GHOSTCOLOUR3; //colour for ghost 3
static COLOURTYPE GHOSTCOLOUR4; //colour for ghost 4
static COLOURTYPE RUNGHOSTCOLOUR; //colour for running ghosts
static COLOURTYPE FOODCOLOUR;   //colour of the food
static COLOURTYPE SUPERFOODCOLOUR; //colour of the superfood
static COLOURTYPE BONUSPOINTCOLOUR;//colour of the bonuspoint bonus
static COLOURTYPE BONUSLIFECOLOUR; //colour of the bonuslife bonus
                           
/*
#define MYFOREGROUND cyan.pixel
#define MYBACKGROUND black.pixel
#define WALLCOLOUR deepskyblue.pixel
#define PACMANCOLOUR yellow.pixel
#define GHOSTCOLOUR1 red.pixel
#define GHOSTCOLOUR2 violetred.pixel
#define GHOSTCOLOUR3 violet.pixel
#define GHOSTCOLOUR4 orangered.pixel
#define RUNGHOSTCOLOUR lightblue.pixel
#define FOODCOLOUR lightyellow.pixel
#define SUPERFOODCOLOUR gold.pixel
#define BONUSPOINTCOLOUR gold.pixel
#define BONUSLIFECOLOUR gold.pixel
*/

};

#endif
