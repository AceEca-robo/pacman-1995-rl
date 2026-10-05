#ifndef __timing_h_
#define __timing_h_

#include "systspec.h"

void printtp(struct timeval);	//write a such struct

#ifdef UNIX                                                                   
void printtms(struct tms);	//write a tms
double difftime(struct timeval,struct timeval);	//difference between two times

#elif defined MSWIN
double difftime(DWORD , DWORD);	//difference between two times
#endif

void timingstart(void);		//initialize timing

void timing(int);		//do timing/synchronization
    
#endif






