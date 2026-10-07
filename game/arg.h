#ifndef __arg_h_
#define __arg_h_
#include"object.h"

class Argument : public Object {

public:

static int argc;	//number of argument	
static char **args;	//pointer to argument strings

static char *rl_socket;	//RL bridge: --rl <path>, 0 if not given
static int fast;	//RL bridge: --fast, timing() does not sleep
static int headless;	//RL bridge: --headless, no X11 at all (needs --rl)
static int no_ghosts;	//RL bridge: --no-ghosts, ghosts stay in their house (diagnostics)
static int has_seed;	//RL bridge: --seed <int> was given
static int seed;	//RL bridge: value of --seed
static int start_level;	//RL bridge: --level <n>, the game starts on level n (needs --rl); 0 = level 1 as always

Argument(int, char **);	//argument constructor

};

#endif
