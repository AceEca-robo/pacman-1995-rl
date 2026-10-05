#include"timing.h"
//#include<sys/wait.h>
#include<stdio.h>
#include"pac.h"


#ifdef UNIX
 static struct timeval *tp1,*tp2;		//current and previous time 
 static struct tms *tmsp;			//time struct
 static double tot_tid;				//time diffenence
 static  clock_t cl;				//clock time
 static int toTid;				//time

#elif defined MSWIN
 static DWORD *tp1,*tp2;			//current and previous time
 static double tot_tid;				//time diffenence
#endif           

#ifdef UNIX      
#include<unistd.h>
      
//write a such struct
void printtp(struct timeval tp) {                     
#ifdef XWIN
 printf("%d s %d u\n",tp.tv_sec,tp.tv_usec);                              
#endif
}            
                                             
//write a tms
void printtms(struct tms t) {                                   
#ifdef XWIN
 double k=.016666666;                                                 
 printf("%f usr, %f sys\n",k*t.tms_utime,k*t.tms_stime);
 printf("%f usr, %f sys : children\n",k*t.tms_cutime,k*t.tms_cstime);
#endif
}          
#endif             
                                                                              
#ifdef UNIX  
 //difference between two times 
 double difftime(struct timeval tp1,struct timeval tp2) {                   
 long a,b;                                                            
 a=tp1.tv_sec-tp2.tv_sec;                           
 b=tp1.tv_usec-tp2.tv_usec;                                
                                                
 if (b<0) { a--; b+= 1000000; }
 return (b*0.000001)+a;                              
}                                                                     

#elif defined MSWIN
double difftime(DWORD tp1,DWORD tp2) {	//difference between two times
 return (tp1-tp2)*0.001;
}
#endif
                                   
void timingstart() {			//initialize timing
#ifdef UNIX       
tp1=new timeval;                  	//allocate structures
tp2=new timeval;               
tmsp=new tms;           
                                        //initialize            
(*tp1).tv_sec=0;                                           
(*tp1).tv_usec=0;                               
                                                      
(*tp2).tv_sec=0;                                      
(*tp2).tv_usec=0;                                                     
                                   
                                                        
gettimeofday(tp1,NULL);             	//get the time                
//printtp(*tp1); 
//printf("start\n"); 
#elif defined MSWIN          
tp1=new DWORD;				//allocate structures
tp2=new DWORD;
*tp1=GetCurrentTime();			//get the time
#endif       
}                                    

// ----------------

void timing(int super){         	//do timing/synchronization

super++;				//divisor
if (!super) pacexit("super zero");	//if would then be division be zero
do {
#ifdef UNIX
 gettimeofday(tp2,NULL);		//get the time now
#elif defined MSWIN
 *tp2=GetCurrentTime();			//get the time now
#endif                                   
 //printtp(*tp2);         
 tot_tid=difftime(*tp2,*tp1);		//compute difference between moments
 if (tot_tid<(0.25/super)) 		//if time interval not yet used up
#ifdef UNIX
 sleep(0);				//sleep
#elif MSWIN
 MSG msg;     
 if (PeekMessage(&msg, NULL, NULL, NULL,PM_NOREMOVE)) 	//if there is msg
  if (GetMessage(&msg, NULL, NULL, NULL)) {		//if message got 
   TranslateMessage(&msg);				//else tranlate and
   DispatchMessage(&msg);				//dispatch the message
  }
#else
 ;					//nothing
#endif 
//if mswin::peek/getmessage                                   
} while (tot_tid<(0.25/super));		//while time interval not used up
//printf("\n\n\n");
//printf("The end. : %f\n",tot_tid);    
#ifdef UNIX
cl=times(tmsp); 			//get the time
#endif           
//printf("%d \n",cl);                                                    
//printtms(*tmsp);                                       
//printf("-----\n");

*tp1=*tp2;     				//current time will be previous time

}


