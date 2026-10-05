#include"usrinter.h"
#include<stdio.h>
#include"gpacman.h"
#include"colour.h"
#include<stdlib.h>
#include"arg.h"
#include"pac.h"
#include<string.h>
#include"board.h"

static GC back;

inp_type UserInterface::in=Keyboard;	//initializers
int UserInterface::x=0;
int UserInterface::y=0;
int UserInterface::last_x=0;
int UserInterface::last_y=0;

UserInterface* UserInterface::_instance=0;

//returns the pointer to the instance
UserInterface* UserInterface::instance(inp_type inp) {
 if (_instance==0) 
  _instance=new UserInterface;
 in=inp;
 return _instance;
}

inline void UserInterface::setinp(inp_type i) { in=i; }

#ifdef XWIN				//for X-Win only
Display* UserInterface::mydisplay=0;	//more initializers
Window UserInterface::mywindow=0;
GC UserInterface::mygc=0;
XEvent UserInterface::myevent={0};
KeySym UserInterface::mykey=0;
XSizeHints UserInterface::myhint={0};
XWMHints UserInterface::mywmhint={0};
int UserInterface::myscreen=0;
unsigned long UserInterface::myforeground=0;
unsigned long UserInterface::mybackground=0;
Font UserInterface::font=0;

#elif defined MSWIN			//for MS-Win only
int UserInterface::quit=0;		//even more initializers
HANDLE UserInterface::my_hInst=0;
HWND UserInterface::my_hwnd=0;    
HFONT UserInterface::hfont=0; 
HFONT UserInterface::hfontOld=0;
int UserInterface::MapModePrevious=0;
HANDLE UserInterface::my_hInstance=0;
HANDLE UserInterface::my_hPrevInstance=0;
LPSTR UserInterface::my_lpCmdLine=0;
int UserInterface::my_nCmdShow=0;
HDC UserInterface::hdcMemory=0;
HBRUSH UserInterface::back=0;
static UserInterface *ui=0;
#endif 
   
#ifdef XWIN				//for X-Win only
UserInterface::UserInterface() {	//constructor
setinp(in);				//sets the input type
mydisplay = XOpenDisplay("");		//connects to a display
if (!mydisplay) pacexit("no display possible");	//if it was not possible, exit
myscreen = DefaultScreen(mydisplay);	//get the default screen
Colour(this);				//initialize the colours
mybackground=Colour::MYBACKGROUND; 	//set background colour
myforeground=Colour::MYFOREGROUND;	//set foreground colour
myhint.x=WINSTARTX; myhint.y=WINSTARTY;	//set window's initial coordinates
myhint.width=WINWIDTH; 			//set window's width and height
myhint.height=WINHEIGHT;
//myhint.width_inc=100;
//myhint.height_inc=100;
//myhint.base_height=WINWIDTH;
//myhint.base_width=WINHEIGHT;
myhint.min_width=WINWIDTH; 			//set window's width and height
myhint.min_height=WINHEIGHT;
myhint.max_width=WINWIDTH; 			//set window's width and height
myhint.max_height=WINHEIGHT;
myhint.flags=PPosition|PSize|PMinSize|PMaxSize;//|PResizeInc|PBaseSize;		//what myhint members were set
mywmhint.flags=InputHint;//|StateHint;		//say we want to have input
mywmhint.input=True;			//that we set inputflag
mywmhint.initial_state=NormalState;
mywindow=XCreateSimpleWindow(mydisplay, DefaultRootWindow(mydisplay),
	     myhint.x,myhint.y,myhint.width,myhint.height,5,
	     myforeground,mybackground);
					//created the window
XSetStandardProperties(mydisplay,mywindow,PACTITLE,PACTITLE,None,Argument::args,Argument::argc,&myhint);
					//some properties were set
XSetWMHints(mydisplay,mywindow,&mywmhint);	//give window manager hints
mygc=XCreateGC(mydisplay,mywindow,0,0);		//create a graphic context
XSetBackground(mydisplay,mygc,mybackground);	//set window background
XSetForeground(mydisplay,mygc,myforeground);	//set window foreground
font=XLoadFont(mydisplay,FONT1);	//load the font
if (!font) font=XLoadFont(mydisplay,FONT2);
XSetFont(mydisplay,mygc,font);		//set the font
XSelectInput(mydisplay,mywindow,ExposureMask);//|VisibilityChangeMask);//,ButtonPressMask|KeyPressMask|ExposureMask|VisibilityChangeMask);
					//select what we want as input
XSetGraphicsExposures(mydisplay,mygc,False);
XMapRaised(mydisplay,mywindow);		//raise the window, that is, show it
do XNextEvent(mydisplay,&myevent);	//wait until window is exposed
while (myevent.type!=Expose);

Window root,child;int dumx,dumy;unsigned int keys_buttons;	//dummies
XQueryPointer(mydisplay,mywindow,&root,&child,&dumx,&dumy,&x,&y,&keys_buttons);
					//get the initial mouse coordinates
XSelectInput(mydisplay,mywindow,ExposureMask|VisibilityChangeMask);//,ButtonPressMask|KeyPressMask|ExposureMask|VisibilityChangeMask);
					//select what we want as input
}

UserInterface::~UserInterface(void) {	//destructor
XUnloadFont(mydisplay,font);		//unload the font
XFreeGC(mydisplay, mygc);		//destroy gc
XDestroyWindow(mydisplay, mywindow);	//destroy window
XCloseDisplay(mydisplay);		//close the display connection
}

void UserInterface::waitsync(void) {
XSync(mydisplay,0);	//returns when output complete: synchronized
}

int UserInterface::getinput(void) {	//get input whether quit or not,
					//it may also change input type
if (QLength(mydisplay)>0) {
XNextEvent(mydisplay, &myevent);	//get next event from the queue
//printf("%d \n",myevent.type);
switch (myevent.type) {			//what kind of event was it?
 case Expose: {
//  Window root,child;int dumx,dumy;unsigned int keys_buttons;	//dummies
//  if (myevent.xexpose.count==0) 	//if this is first expose
//   XQueryPointer(mydisplay,mywindow,&root,&child,&dumx,&dumy,&x,&y,&keys_buttons);
//					//get the initial mouse coordinates
// XRaiseWindow(mydisplay,mywindow);	//let window stay on top
#ifdef XPOSSE
 writetext();
 Board::instance()->draw();
// printf("XPOSSE\n");
 waitsync(); 
#endif
 }
 break;
#ifdef RAISE
 case VisibilityNotify:
  if (myevent.xvisibility.state!=VisibilityUnobscured) 
   XRaiseWindow(mydisplay,mywindow);	//let window stay on top
 break;
#endif
 case MappingNotify:			//something about the keyboard mapping
  XRefreshKeyboardMapping(&myevent.xmapping);
 break;
 case MapNotify:
//   XRaiseWindow(mydisplay,mywindow);	//let window stay on top
 break;
 }

}

if (in==Mouse) {			//if input is from mouse
 Window root,child;int dumx,dumy,xx,yy; //dummies
 unsigned int keys_buttons;		//what keys/buttons hit/pressed
 XQueryPointer(mydisplay,mywindow,&root,&child,&dumx,&dumy,&xx,&yy,&keys_buttons);
 if (keys_buttons & Button1Mask) 	//if left mouse button down 
  in=Keyboard;				//input type is keyboard
 if ((keys_buttons & Button2Mask) || (keys_buttons & Button3Mask)) 
  return 1;				//if middle or right is down quit
} else {				//else if input is from keyboard
 char key_vector[32];			//denotes 256 possible keys
 XQueryKeymap(mydisplay,key_vector);	//queries and gets data for all keys
//int kkk; for (kkk=0;kkk<32;kkk++)
//printf("%x ",key_vector[kkk]);
//printf("\n");
 if (check(key_vector,XK_m))		//if m is pressed input type is mouse
  { in=Mouse; return 0; }
 if (check(key_vector,XK_M))		//if m is pressed input is keyboard
  { in=Mouse; return 0; }
 if (check(key_vector,XK_Q)) return 1;	//if q is pressed quit
 if (check(key_vector,XK_q)) return 1;	//if q is pressed quit
}
//}
return 0;
}

//checks whether certain key is pressed
int UserInterface::check(char key_vector[32], KeySym key) {
unsigned int byteindex, bitindex;	//byte and bit number 
KeyCode code;				//keykode
code=XKeysymToKeycode(mydisplay,key);	//get the code of the wanted key
if (code==NoSymbol) pacexit("NoSymbol\n");	//if no such, exit
byteindex=(code/8);				//what's it's index number
bitindex=code & 7;				//what's it's bit number
return (1 & (key_vector[byteindex]>>bitindex));	//has it been hit?
}

direction UserInterface::stick(void) {
if (in==Mouse) {						//mouse input
Window root,child;int dumx,dumy; unsigned int keys_buttons;	//dummies
 last_x=x; last_y=y;					//present is past
XQueryPointer(mydisplay,mywindow,&root,&child,&dumx,&dumy,&x,&y,&keys_buttons);
						//get new coordinates
if (x==last_x && y==last_y) return none;	//if not moved then none
int dx=x-last_x;int dy=y-last_y;		//get changes of coordinates
if (dx>=dy && dx>-dy) return right;		//destine directions
if (dx>=dy && dx<=-dy) return up;
if (dx<dy && dx>-dy) return down;
if (dx<dy && dx<=-dy) return left;
pacwarning("errorstick\n");	//give warning if not all directons covered
} else {			//if input comes from keyboard
char key_vector[32];		//denotes 256 possible keys
XQueryKeymap(mydisplay,key_vector);	//queries and gets data for all keys
if (check(key_vector,XK_Up)) return up;		//if up key pressed, up
if (check(key_vector,XK_Down)) return down;	//etc
if (check(key_vector,XK_Left)) return left;
if (check(key_vector,XK_Right)) return right;
return none;					//if none of these, then none
}
}

//write string at x,y
void UserInterface::write(int x,int y,char* str) {
 XDrawImageString(mydisplay,mywindow, mygc,TEXTWIDTH*x,TEXTHEIGHT*y,str, strlen(str));
}

//write number long int at x,y format z
void UserInterface::write(int x,int y,long int v,int l) {
 char*  str;
 str = (char*) malloc (l+1);
 sprintf (str, "%.*d", l,v);
 XDrawImageString(mydisplay,mywindow, mygc,TEXTWIDTH*x,TEXTHEIGHT*y,str, strlen(str));
}

#elif defined MSWIN					//for MS-Win only
void UserInterface::sethInstance(HANDLE h) {my_hInstance=h; }	//setters
void UserInterface::sethPrevInstance(HANDLE h) {my_hPrevInstance=h; }
void UserInterface::setlpCmdLine(LPSTR l) {my_lpCmdLine=l; }
void UserInterface::setnCmdShow(int i) {my_nCmdShow=i; }
HANDLE UserInterface::gethInstance(void) {return my_hInstance; }//getters
HANDLE UserInterface::gethPrevInstance(void) {return my_hPrevInstance; }
LPSTR UserInterface::getlpCmdLine(void) {return my_lpCmdLine; }
int UserInterface::getnCmdShow(void) {return my_nCmdShow; }
HWND UserInterface::gethwnd(void) {return my_hwnd;}
HDC UserInterface::gethdcmem(void) {return hdcMemory; }

UserInterface::UserInterface() {	//constructor
HDC my_hdc,				//handle to device context
    my_hdcMemory;			//handle to memory device context
setinp(in);				//sets input type
Colour(this);   			//initializing colours
ui=this;				//set userinterface ui variable
 if (!my_hPrevInstance)			//if no previous application instance
  if (!InitApplication(my_hInstance))	//then try to initialize application
   pacexit("FALSE");			//if error do an exit
 if (!InitInstance(my_hInstance, my_nCmdShow))	//try to initialize an instance
  pacexit("FALSE");   			//if error do an exit
EnableWindow(my_hwnd,TRUE);		//enables mouse/keyboard input
my_hdc=GetDC(my_hwnd); 			//get the device context
if (!my_hdc) {				//if none such then exit
 exit(1);
}           
hdcMemory = CreateCompatibleDC(my_hdc);	//create a compatible device context
int iPtSize = TEXTHEIGHT;		//specify height of text
PSTR pszFace = 0;//*"MS Serif"; */;	//default face/font
MapModePrevious = SetMapMode(my_hdc, MM_TEXT);	//setting of mapping mode
hfont = CreateFont(-iPtSize, TEXTWIDTH, 0, 0, 0, /* specify pt size    */
    0, 0, 0, 0, 0, 0, 0, FF_MODERN | FIXED_PITCH, pszFace); //creates a font
hfontOld = SelectObject(my_hdc, hfont);	//selects a font and returns old
SetBkMode(my_hdc,OPAQUE);   		//set background mode
SetBkColor(my_hdc,Colour::MYBACKGROUND);	//set background colour
SetTextColor(my_hdc,Colour::MYFOREGROUND);	//sets text colour
MoveTo(my_hdc,0,0);			//move to coordinates 0,0
ReleaseDC(my_hwnd,my_hdc);  		//release the device context
}

UserInterface::~UserInterface(void) {
HDC my_hdc,my_hdcMemory;		//for ordinary and memory dc
my_hdc=GetDC(my_hwnd);			//get the device context
SetMapMode(my_hdc, MapModePrevious);	//set mapping mode to the previous
SelectObject(my_hdc, hfontOld);		//select back the old font
DeleteObject(hfont); 			//delete the games font
ReleaseDC(my_hwnd,my_hdc);		//release the device context
DeleteObject(back);			//delete the background colour brush
}

void UserInterface::waitsync(void) {	//returns when output synchronized
}					//not necessary here, automatic

direction UserInterface::stick(void) {	//returns the inputted direction
if (in==Mouse) {			//if input is from mouse
last_x=x; last_y=y;     		//present is past
POINT p; 				//a point
GetCursorPos(&p);			//get position of mouse cursor
x=p.x; y=p.y;				//set current coordinates to these
if (x==last_x && y==last_y) return none;//if not moved then none
int dx=x-last_x; int dy=y-last_y;	//get changes of coordinates
if (dx>=dy && dx>-dy) return right;	//destine directions
if (dx>=dy && dx<=-dy) return up;
if (dx<dy && dx>-dy) return down;
if (dx<dy && dx<=-dy) return left;
pacwarning("errorstick\n");	//give warning if not all directons covered
} else {			//if input comes from keyboard
if (GetAsyncKeyState(VK_UP)<0) return up;	//if up key pressed, up
if (GetAsyncKeyState(VK_DOWN)<0) return down;	//etc
if (GetAsyncKeyState(VK_LEFT)<0) return left;
if (GetAsyncKeyState(VK_RIGHT)<0) return right;
return none;					//if none of these, then none
}                   
return none;					//if none of these, then none
}

//write string at x,y
void UserInterface::write(int x,int y,char* str) {
HDC my_hdc,my_hdcMemory;		//for ordinary and memory dc
my_hdc=GetDC(my_hwnd);			//get device context
 TextOut(my_hdc,TEXTWIDTH*x,TEXTHEIGHT*y,str, strlen(str));	//outputting
ReleaseDC(my_hwnd,my_hdc);  		//release device context
}

//write number long int at x,y format z
void UserInterface::write(int x,int y,long int v,int l) {
 HDC my_hdc,my_hdcMemory;		//for ordinary and memory dc
 char*  str;				//string
 my_hdc=GetDC(my_hwnd);			//get device context
 str = (char*) malloc (l+1);		//allocate memory for string
 sprintf (str, "%.*d", l,v);   		//formatted print to string
 TextOut(my_hdc,TEXTHEIGHT*x,TEXTHEIGHT*y,str, strlen(str));	//outputting
 free(str);							//free memory
 ReleaseDC(my_hwnd,my_hdc);  		//release device context
}
     
int UserInterface::getinput(void) {	//get input whether quit or not,
					//it may also change input type
 MSG msg;    
 if (!PeekMessage(&msg, NULL, NULL, NULL,PM_NOREMOVE))
  return 0;			//just return if there wasn't a message
 quit=!GetMessage(&msg, NULL, NULL, NULL);	//quit if GetMessage says so
 if (quit) return 1;   				//do the quit
 TranslateMessage(&msg);			//else tranlate and
 DispatchMessage(&msg);				//dispatch the message
						//control now goes to
						//a WNDPROC MainWndProc
 return quit;				//return whether quit
}

void UserInterface::do_inp(char wParam) {	//processes input from keyboard
 if (in==Keyboard) {
  if (wParam=='M')  in=Mouse; 		//if m pressed input type is now mouse
  if (wParam=='Q')  quit=1;		//if q is pressed then set quit
 }       
}

void UserInterface::do_inp(char *s) {		//processes input from mouse
 if (in==Mouse) {
  if (!strcmp(s,"LEFT")) 
   ui->setinp(Keyboard);  		//if left button input type is keyboard
  if (!strcmp(s,"RIGHT")) quit=1;	//if right butten, set quit	
 }
}

void UserInterface::start(int xx,int yy) {
 x=xx;   y=yy;			//setting current mouse coordinates and telling
 SetFocus(my_hwnd);		//MS-Win to take input from our window
}

void UserInterface::setquit(void) { quit=1; }	//set quit
	       
static long FAR PASCAL MainWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
 switch (message)    {		//what type of message it is
  case WM_CREATE: {		// message: create window
   POINT p;			//a point
   GetCursorPos(&p);		//get position of mouse cursor
   ui->start(p.x,p.y);		//initialize userinterface with that point
  } break;
  case WM_LBUTTONDOWN:  {	// message: left mouse button pressed 
   ui->do_inp("LEFT");		//process code for left mouse button
  }  break;
  case WM_RBUTTONDOWN: {	// message: right mouse button pressed
   ui->do_inp("RIGHT");		//process code for left mouse button
  } break;
  case WM_KEYDOWN : {		// message: keys pressed
   ui->do_inp(wParam);		//process code for keyboard keys
  } break;
  case WM_DESTROY: {  		// message: destroy from window manager
   ui->setquit();		//set quit
   PostQuitMessage(0); }	//post a message that says we quit
  break;
  default:
   return (DefWindowProc(hWnd, message, wParam, lParam));
				//message must be processed
 }
 return (NULL);			//or whatever, just a return value
}

/****************************************************************************

    FUNCTION: InitApplication(HANDLE)

    PURPOSE: Initializes window data and registers window class

****************************************************************************/

BOOL UserInterface::InitApplication(HANDLE hInstance)
{
    WNDCLASS  wc;				//a window class (not oo)
    back=CreateSolidBrush(Colour::MYBACKGROUND);//background brush
    wc.style = CS_BYTEALIGNWINDOW | CS_CLASSDC | CS_VREDRAW | CS_HREDRAW;
    wc.lpfnWndProc = MainWndProc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hInstance;
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = back;/*COLOR_BACKGROUND+1;*/
    wc.lpszMenuName =  "PacmanMenu";		//name
    wc.lpszClassName = "PacmanWClass";		//name
    return (RegisterClass(&wc));		//try to register and
						//return success boolean
}


/****************************************************************************

    FUNCTION:  InitInstance(HANDLE, int)

    PURPOSE:  Saves instance handle and creates main window

****************************************************************************/

BOOL UserInterface::InitInstance(HANDLE hInstance, int nCmdShow)
{
    HWND            hwnd;			//handle to window

    my_hInst = hInstance;			//sets handle to instance

    hwnd = CreateWindow("PacmanWClass",
			PACTITLE,
			WS_OVERLAPPEDWINDOW,
			WINSTARTX,
			WINSTARTY,
			WINWIDTH,
			WINHEIGHT,
			NULL,
			NULL,
			hInstance,
			NULL
		       );			//creates the window
    if (!hwnd)					//if no handle return error
	return (FALSE);
    ShowWindow(hwnd, nCmdShow);			//show and
    UpdateWindow(hwnd);				//update window
    my_hwnd=hwnd;				//set the handle
    return (TRUE);				//return ok
}

#elif defined VTX			//vt??? unix only

UserInterface::UserInterface() {	//constructor
setinp(in=Keyboard);			//set input type
stdscr=initscr();			//initialize screen
raw();					//an input mode
cbreak();				//set to break mode
noecho();				//don't cho to screen
nonl();					//no newline
intrflush(stdscr,FALSE);		//flushing mode
keypad(stdscr,TRUE);			//keypadding
nodelay(stdscr,TRUE);			//no input delay
clear();				//clear screen
Colour(this);				//initialize colours
}

UserInterface::~UserInterface(void) {	//destructor
clear();				//clear screen
refresh();				//refresh it
endwin();				//end it
}

void UserInterface::waitsync(void) {	//returns when output completed
move(0,0);				//move to 0,0
refresh();				//refresh screen
}

int UserInterface::getinput(void) {	//get input whether quit or not,
					//it may also change input type
int c;
(c=(getch())); 
 if (c=='Q' || c=='q') return 1;	//if q pressed then do quit
return 0;
}

direction UserInterface::stick(void) {	//get inputted direction
int key=0,i;
key=getch();				//get input character
if (key==/*ACS_UARROW*/KEY_UP) return up;	//if up key pressed, up
if (key==/*ACS_DARROW*/KEY_DOWN) return down;	//etc
if (key==/*ACS_LARROW*/KEY_LEFT) return left;
if (key==/*ACS_RARROW*/KEY_RIGHT) return right;
return none;					//if none of these, then none
}

//write string at x,y
void UserInterface::write(int x,int y,char* str) {
 mvaddstr(TEXTHEIGHT*y,TEXTWIDTH*x,str);//outputting
}

//write number long int at x,y format z
void UserInterface::write(int x,int y,long int v,int l) {
 char*  str;				//string
 str = (char*) malloc (l+1);		//allocate memory for string
 sprintf (str, "%.*d", l,v);		//formatted print to string
 mvaddstr(TEXTHEIGHT*y,TEXTWIDTH*x,str);//outputting
}

#endif
       
