#include"arg.h"
#include<stdlib.h>	// RL bridge
#include<string.h>	// RL bridge
#include<stdio.h>	// RL bridge

int Argument::argc=0;	//resetters
char** Argument::args=0;

char* Argument::rl_socket=0;	// RL bridge
int Argument::fast=0;		// RL bridge
int Argument::headless=0;	// RL bridge
int Argument::has_seed=0;	// RL bridge
int Argument::seed=0;		// RL bridge

Argument::Argument(int c, char **s) {	//initializes;
// RL bridge: take out --rl/--fast/--seed so that the remaining arguments
// (colour.cc treats any argument as "grey" unless "colour" given) are
// exactly what the original game would have seen
char **rest=new char*[c+1];	// RL bridge
int n=0,i;			// RL bridge
for(i=0;i<c;i++) {		// RL bridge
 if (i>0 && !strcmp(s[i],"--rl") && i+1<c) rl_socket=s[++i];
 else if (i>0 && !strcmp(s[i],"--fast")) fast=1;
 else if (i>0 && !strcmp(s[i],"--headless")) headless=1;
 else if (i>0 && !strcmp(s[i],"--seed") && i+1<c) { has_seed=1; seed=atoi(s[++i]); }
 else rest[n++]=s[i];
}
rest[n]=0;			// RL bridge
argc=n;				// RL bridge
args=rest;			// RL bridge
if (headless && !rl_socket) {	// RL bridge: no keyboard without X
 fprintf(stderr,"--headless needs --rl\n"); exit(1);
}
}
