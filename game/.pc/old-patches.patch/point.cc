#include"point.h"
Point::Point(int xx=0,int yy=0) { x=xx; y=yy; }; 
Point::val_x() { return x; };
Point::val_y() { return y; };
void Point::set_x(int xx) { x=xx; };
void Point::set_y(int yy) { y=yy; };
void Point::set_xy(int xx,int yy) { x=xx; y=yy; };
