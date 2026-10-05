#ifndef __point_h_ 
#define __point_h_
 #include"object.h"
 
class Point : public Object {

int x,y;

public:

Point(int,int);
int val_x(void);
int val_y(void);
void set_x(int x);
void set_y(int y);
void set_xy(int x,int y);

//Point& operator=(const Point& from) {
//this->x=from.x;
//this->y=from.y;
//}

};

#endif                  
