#ifndef __rlbridge_h_
#define __rlbridge_h_
// RL bridge: connection between the game and an RL agent over a Unix socket.
// The game is the client; one JSON line of state goes out per tick, one
// line with an action (U/D/L/R/N) comes back.

#include"direc.h"

class Pacman;
class Ghost;
class Gamedata;
class Bonus;

int rl_active(void);		//true if --rl was given and socket still open
void rl_connect(const char *);	//connect to the socket, exit on failure
void rl_send_state(Pacman *, Ghost **, Gamedata *, Bonus *);	//one JSON line
direction rl_recv_action(void);	//blocking; none on "N" or disconnect
int rl_disconnected(void);	//true once the peer has gone away

#endif
