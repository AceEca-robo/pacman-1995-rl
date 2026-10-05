#ifndef __graphele_h_
#define __graphele_h_

#include"systspec.h"
#include"object.h"
#include"usrinter.h"
#include"shapes.h"
#include"colour.h"
#include"types.h"

//decide the pixmap type
#ifdef XWIN
#define PIXMAPTYPE Pixmap
#elif defined MSWIN
#define PIXMAPTYPE HBITMAP
#elif defined VTX
#define PIXMAPTYPE char
#endif 
 
class GraphElement : public Object {
 protected:

#ifdef XWIN				//for X-Win only
Display *display;			//pointer to display connection
Window window;				//the window itself
int screen;				//the screen
GC gc;					//the graphic context
unsigned int depth;			//the screen depth 

#elif defined MSWIN			//for MS-Win only
HWND hwnd;				//handle to the window itself
HDC hdcMemory;				//handle to the device context memory
#endif
 
PIXMAPTYPE pixmap; 			//this is the pixmap
 
void consfn(); 				//function to help with constructing
void pix(PIXMAPTYPE *,SHAPETYPE [],COLOURTYPE,COLOURTYPE);	//create pixmap

public:

virtual void draw(int,int);		//draw graphical element
virtual ~GraphElement(void)=0;        	//destructor
};

#endif 

