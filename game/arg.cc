#include"arg.h"

int Argument::argc=0;	//resetters
char** Argument::args=0;

Argument::Argument(int c, char **s) {	//initializes;
argc=c;
args=s;
}
