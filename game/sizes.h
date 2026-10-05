#include"basis.h"

#define BOARDWIDTH 33
#define BOARDHEIGHT 23

#ifndef VTX		//if not vt??? unix
#if defined(MSWIN) || !defined(DOUBLE)
#define UNITWIDTH 8		//pixmaps is 8x8
#define UNITHEIGHT 8
#else 
#define UNITWIDTH 16 
#define UNITHEIGHT 16
#endif
#else			//if vt??? unix
#define UNITWIDTH 1	//character 1x1 contains "pixmaps"
#define UNITHEIGHT 1
#endif

//defines different sizes for different fonts
#ifdef XWIN
#ifndef DOUBLE
#define TEXTWIDTH 8
#define TEXTHEIGHT 8
#define FONT1 "5x8"
#define FONT2 "5x7"
#else
#define TEXTWIDTH 8
#define TEXTHEIGHT 16
#define FONT1 "8x16"
#define FONT2 "7x14"
#endif
//#define PFONT "clR6x8.fb"/*clR8x8" "6x10"*/
#elif defined MSWIN
#define TEXTWIDTH 6
#define TEXTHEIGHT 10
//#define PFONT 0
#elif defined VTX
#define TEXTWIDTH 1	//character 1x1 contain lots
#define TEXTHEIGHT 1
#endif

#define TEXTCHARSINLINE 15	//text characters befors board begins

#define TEXTSPACE TEXTWIDTH*TEXTCHARSINLINE		
#define WINWIDTH BOARDWIDTH*UNITWIDTH+TEXTSPACE+20
#define WINHEIGHT BOARDHEIGHT*UNITHEIGHT+40



