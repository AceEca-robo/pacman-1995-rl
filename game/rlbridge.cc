// RL bridge: see rlbridge.h
#include"rlbridge.h"
#include"pac.h"
#include"pacman.h"
#include"ghost.h"
#include"gamedata.h"
#include"board.h"
#include"bonus.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<errno.h>
#include<sys/socket.h>
#include<sys/un.h>

static int fd=-1;		//socket, -1 if not connected
static int lost=0;		//peer went away
static char inbuf[256];		//partial input line
static int inlen=0;

int rl_active(void) { return fd>=0; }

int rl_disconnected(void) { return lost; }

static void rl_close(void) {	//peer gone: stop talking, let the game end
 if (fd>=0) close(fd);
 fd=-1;
 lost=1;
}

void rl_connect(const char *path) {
 struct sockaddr_un a;
 if (strlen(path)>=sizeof(a.sun_path)) { fprintf(stderr,"rl: socket path too long\n"); exit(1); }
 fd=socket(AF_UNIX,SOCK_STREAM,0);
 if (fd<0) { perror("rl: socket"); exit(1); }
 memset(&a,0,sizeof(a));
 a.sun_family=AF_UNIX;
 strcpy(a.sun_path,path);
 if (connect(fd,(struct sockaddr*)&a,sizeof(a))<0) { perror("rl: connect"); exit(1); }
}

static void sendall(const char *s,int n) {
 while (fd>=0 && n>0) {
  int k=send(fd,s,n,MSG_NOSIGNAL);
  if (k<0 && errno==EINTR) continue;
  if (k<=0) { rl_close(); return; }
  s+=k; n-=k;
 }
}

static char dirchar(direction d) {
 switch (d) {
  case up: return 'U';
  case down: return 'D';
  case left: return 'L';
  case right: return 'R';
  default: return 'S';		//still (or none): not moving
 }
}

static char cellchar(typ t) {
 switch (t) {
  case classWall: return '#';
  case classFood: return '.';
  case classSuperFood: return 'o';
  case classSpecialWall: return '-';	//the ghost house gate
  default: return ' ';
 }
}

void rl_send_state(Pacman *pac,Ghost **gh,Gamedata *da,Bonus *bon) {
 if (fd<0) return;
 // grid 23*(33+3) + entities: well below 4k
 static char buf[8192];
 int n=0,i,j,x,y;
 Board *b=Board::instance();

 n+=sprintf(buf+n,"{\"grid\":[");
 for(j=0;j<BOARDHEIGHT;j++) {
  buf[n++]='"';
  for(i=0;i<BOARDWIDTH;i++) buf[n++]=cellchar(b->what_is(i,j));
  buf[n++]='"';
  if (j<BOARDHEIGHT-1) buf[n++]=',';
 }
 pac->getxy(&x,&y);
 n+=sprintf(buf+n,"],\"pacman\":{\"x\":%d,\"y\":%d,\"dir\":\"%c\",\"try_dir\":\"%c\"},\"ghosts\":[",
            x,y,dirchar(pac->getdir()),dirchar(pac->gettrydir()));
 for(i=0;i<GHOSTS;i++) {
  const char *st;
  switch (gh[i]->getstate()) {
   case hunted: st="hunted"; break;
   case eaten: st="eyes"; break;
   default: st="normal"; break;	//randm and hunter
  }
  gh[i]->getxy(&x,&y);
  n+=sprintf(buf+n,"%s{\"x\":%d,\"y\":%d,\"dir\":\"%c\",\"state\":\"%s\"}",
             i?",":"",x,y,dirchar(gh[i]->getdir()),st);
 }
 n+=sprintf(buf+n,"],\"bonus\":");
 if (bon) {
  bon->getxy(&x,&y);
  n+=sprintf(buf+n,"{\"x\":%d,\"y\":%d,\"type\":\"%s\"}",x,y,
             bon->is_a()==classBonusLife?"life":"point");
 } else n+=sprintf(buf+n,"null");
 n+=sprintf(buf+n,",\"score\":%ld,\"lives\":%d,\"level\":%d,"
            "\"supertime_left\":%d,\"done\":%s}\n",
            da->getscore(),da->getlives(),da->getlevel(),
            pac->is_super()?pac->getsupertime():0,
            da->getlives()?"false":"true");
 sendall(buf,n);
}

direction rl_recv_action(void) {
 for(;;) {
  char *nl=(char*)memchr(inbuf,'\n',inlen);
  if (nl) {
   char c=inbuf[0];
   int used=nl-inbuf+1;
   memmove(inbuf,inbuf+used,inlen-used);
   inlen-=used;
   switch (c) {
    case 'U': return up;
    case 'D': return down;
    case 'L': return left;
    case 'R': return right;
    default: return none;	//N (or anything else): keep direction
   }
  }
  if (fd<0) return none;
  if (inlen==sizeof(inbuf)) inlen=0;	//garbage without newline: drop it
  int k=recv(fd,inbuf+inlen,sizeof(inbuf)-inlen,0);
  if (k<0 && errno==EINTR) continue;
  if (k<=0) { rl_close(); return none; }
  inlen+=k;
 }
}
