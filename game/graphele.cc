#include"graphele.h"
#include<stdio.h>
#include<stdlib.h>

#ifdef XWIN				//for X-Win only
void GraphElement::draw(int x,int y) {	//draw graphical element
XCopyArea(display,pixmap,window,gc,0,0,UNITWIDTH,UNITHEIGHT,TEXTSPACE+UNITWIDTH*x,UNITWIDTH*y);				//copy pixmap to window
 }

void GraphElement::consfn() {		//function to help with constructing
UserInterface* u=UserInterface::instance();	//get UserInterface instance
display=u->getdisplay();		//get pointer to display connection
window=u->getwindow();			//get window
screen=u->getscreen();			//get screen
gc=u->getgc();				//get graphic context
int wx,wy;				//dummies
Window root;				//dummy
unsigned int wwidth,wheight,wdepth,wborder,width,height,status;
					//dummies, except wdepth
XGetGeometry(display,window,&root,&wx,&wy,&wwidth,&wheight,
  &wborder,&wdepth);			//get X data
depth=wdepth;				//setting depth
}

GraphElement::~GraphElement(void) {	//destructor
XFreePixmap(display,pixmap);		//freeing pixmap
}

//create pixmap
void GraphElement::pix(Pixmap *p,char map[],unsigned long myforeground, unsigned long mybackground) {
*p=XCreatePixmapFromBitmapData(display,window,map,UNITWIDTH,UNITHEIGHT,
				   myforeground,mybackground,depth);
}
  
#elif defined MSWIN			//for MS-Win only  
void GraphElement::draw(int x,int y) {	//draw graphical element
HDC hdcDest;				//destination device context
HBITMAP hbmpOld;			//previously selected bitmap
hdcDest=GetDC(hwnd); 			//select device context
hbmpOld=SelectObject(hdcMemory,pixmap);	//select our pixmap
BitBlt(hdcDest,TEXTSPACE+UNITWIDTH*x,UNITHEIGHT*y,UNITWIDTH,UNITHEIGHT,hdcMemory,0,0, SRCCOPY); 			//copy pixmap to window
SelectObject(hdcMemory, hbmpOld);	//select back previous bitmap
ReleaseDC(hwnd,hdcDest);  		//release device context
}

void GraphElement::consfn() {		//function to help with constructing
UserInterface* u=UserInterface::instance();	//get userinterface instance
HDC hdcDest;				//destination device context
hwnd=u->gethwnd();  			//get handle to window
hdcMemory = u->gethdcmem();		//get and set device context memory
//cbBits=16;
}

GraphElement::~GraphElement(void) {	//destructor
DeleteObject(pixmap);			//delete the pixmap
}

void GraphElement::pix(HBITMAP *p,short map[],COLOURNAME foreground, COLOURNAME background) {				//create pixmap
    HDC hdcDest;			//destination device context
    HDC hDC;				//device context
    HDC hMemoryDC;			//memory device context
    HBITMAP hBitmap;			//current bitmap
    HBITMAP hOldBitmap;			//previous bitmap
    HBRUSH hForeBrush;			//foreground colour brush
    HBRUSH hBackBrush;			//background colour brush
    HBRUSH hOldBrush;			//previous brush
    hDC = GetDC(hwnd);			//get device context for window
    hMemoryDC = CreateCompatibleDC(hDC);//create a compatible device context
    hBitmap = CreateCompatibleBitmap(hDC, UNITWIDTH, UNITHEIGHT);
					//create a compatible bitmap
    hOldBitmap = SelectObject(hMemoryDC, hBitmap);	//select our bitmap
    hForeBrush = CreateSolidBrush(foreground);		//create fore brush
    hBackBrush = CreateSolidBrush(background);		//create back brush
    hOldBrush = SelectObject(hMemoryDC, hForeBrush);	//select fore brush
    PatBlt(hMemoryDC, 0, 0, UNITWIDTH,UNITHEIGHT, BLACKNESS);
					//set pixmap to all blackness
int x,y,				//counters
 bitno=0;				//bit number
short* mapptr=map;			//initial source counter
short byte=*mapptr;			//initial content
for(y=0; y<UNITHEIGHT; y++)		//traverse entire pixmap bit for bit
 for(x=0; x<UNITWIDTH; x++) {
  if (byte & 0x80)			//if leftmost bit is 1
   SetPixel(hMemoryDC,x,y,foreground);
//   SelectObject(hMemoryDC, hForeBrush);	//then select fore brush
   SetPixel(hMemoryDC,x,y,background);
  else					//if leftmost bit is 0
//   SelectObject(hMemoryDC, hBackBrush);	//then select back brush
//  PatBlt(hMemoryDC, x, y, 1, 1, PATCOPY);	//set bit at x,y to that brush
  bitno++;				//increment bit number
  if (bitno==8) {			//if byte is "used up"
   bitno=0;				//start again: bit number = 0 and
   mapptr++; 				//increment map counter
   byte=*mapptr;			//new content
  } else 				//if byte is not "used up"
   byte<<=1;				//shift content left 
}
    SelectObject(hMemoryDC, hOldBrush);	//select back previous brush
    DeleteObject(hForeBrush);		//delete fore brush
    DeleteObject(hBackBrush);		//delete back brush
    SelectObject(hMemoryDC, hOldBitmap);//select back previous bitmap
    DeleteDC(hMemoryDC);		//delete memory device context
    ReleaseDC(hwnd, hDC);		//release device context
*p=hBitmap;				//"return" the bitmap
}

#elif defined VTX			//vt??? unix only
void GraphElement::draw(int x,int y) {	//draw graphical element
mvaddch(UNITWIDTH*y,TEXTSPACE+UNITWIDTH*x,pixmap);
}

void GraphElement::consfn() {		//function to help with construction
}					//not really necessery here

GraphElement::~GraphElement(void) {	//destructor
}					//not really necessery here

void GraphElement::pix(PIXMAPTYPE *p,SHAPETYPE map[],COLOURTYPE myforeground, COLOURTYPE mybackground) {		//create pixmap
*p=*map;				//"return" it
}
#endif
