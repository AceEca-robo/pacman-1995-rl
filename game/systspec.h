#include"basis.h"

#ifdef XWIN
 #include<X11/Xlib.h>
 #include<X11/Xutil.h>
 #include<X11/keysym.h>
//#elif defined MAC
// #include<console.h>
// #include<cwindows.h>
// #include<cbitmap.h>
#elif defined MSWIN
 #include<windows.h>
#elif defined VTX
// #ifndef MYCURSES_H
// #define MYCURSES_H
 #include<curses.h>
// #endif
#else
 #error UNKNOWN TYPE DEFINED
#endif

#ifdef UNIX 
 #include <sys/time.h>
 #include <sys/times.h>
#endif
