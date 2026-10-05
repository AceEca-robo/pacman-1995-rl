#ifndef __usrinter_h_
#define __usrinter_h_ 
#include"object.h"
#include"sizes.h"
#include"direc.h"

#include"systspec.h"

#define WINSTARTX 200
#define WINSTARTY 300

//is the input from mouse or keyboard
typedef enum {Mouse,Keyboard} inp_type;

//here is the graphical id type defined
#ifdef XWIN
#define GID_TYPE Pixmap 
#elif defined MSWIN
#define GID_TYPE HBITMAP
#elif defined VTX
#define GID_TYPE char
#endif

class UserInterface : public Object {

static UserInterface* _instance;//pointer to the instance of this class

static inp_type in;		//is the input from mouse or keyboard?

static int x,y,			//current mouse coordinates
       last_x,last_y;		//previous mouse coordinates

#ifdef XWIN			//for X-Win only
static Display *mydisplay;	//pointer to display connection
static Window mywindow;		//the window itself
static GC mygc;			//the graphic context
static XEvent myevent;		//an event
static KeySym mykey;		//a key pressed
static XSizeHints myhint;	//hints about sizes in X
static XWMHints mywmhint;	//hints to the X Window Manager
static int myscreen;		//the screen
static unsigned long myforeground,	//foreground and
		     mybackground;	//background colour in window
static Font font;			//font used
static int check(char *,KeySym);	//checks whether certain key is pressed

#elif defined MSWIN		//for MS-Win only
static int quit;		//shall application quit?
static HANDLE my_hInst;		//handle to this instance of the application
static HWND my_hwnd;    	//handle to the window itself
static HFONT hfont, hfontOld;	//current and previous font
static int MapModePrevious;	//what's the previous mapping mode.
static HANDLE my_hInstance;	//handle to this instance of the application
static HANDLE my_hPrevInstance;	//handle to previous instance of application
static LPSTR my_lpCmdLine;	//command line give
static int my_nCmdShow;		//about showing command?
static BOOL InitApplication(HANDLE);	//initializing the application
static BOOL InitInstance(HANDLE, int);	//init. of this instance of application
static HDC hdcMemory;		//handle to the device context memory
static HBRUSH back;		//the brush for the background colour
#endif

 protected:

UserInterface();		//constructor

public:

static UserInterface* instance(inp_type = Keyboard);
				//returns the pointer to the instance,
				//with keyboard as default 
~UserInterface(void); 		//destructor

#ifdef XWIN			//for MS-Win only
static Display* getdisplay() 	//all those are returning corresponding values
	{return mydisplay;}
static Window getwindow() 
	{return mywindow;}
static int getscreen() 
	{return myscreen;}
static GC getgc() 
	{return mygc;}
static XEvent getevent() 
	{return myevent;}

#elif defined MSWIN		//for MS-Win only
static HANDLE gethInst(void);	//all those are returning corresponding values
static HWND gethwnd(void);
static HDC gethdcmem(void);
static HANDLE gethInstance(void);
static HANDLE gethPrevInstance(void);
static LPSTR getlpCmdLine(void);
static int getnCmdShow(void);
static void sethInst(HANDLE);	//all those are setting corresponding values
static void sethwnd(HWND); 
static void sethInstance(HANDLE);
static void sethPrevInstance(HANDLE);
static void setlpCmdLine(LPSTR);
static void setnCmdShow(int);
static void start(int,int);	//setting current mouse coordinates and telling
				//MS-Win to take input from our window
static void do_inp(char);	//processes input from keyboard
static void do_inp(char *);	//processes input from mouse
static void setquit(void);	//set quit true

#endif

static direction stick(void);	//returns the inputted direction

static void write(int,int,char*);	//write string at x,y
static void write(int,int,long int,int);//write number long int at x,y format z

static int getinput(void);	//return whether quit or not
static void waitsync(void);	//returns when output complete: synchronized
static void setinp(inp_type);	//sets the input type
};

#endif
