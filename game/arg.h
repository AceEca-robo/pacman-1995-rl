#ifndef __arg_h_
#define __arg_h_
#include"object.h"

class Argument : public Object {

public:

static int argc;	//number of argument	
static char **args;	//pointer to argument strings

Argument(int, char **);	//argument constructor

};

#endif

