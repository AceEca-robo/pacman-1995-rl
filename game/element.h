#ifndef __element_h_
#define __element_h_

#include"object.h"
#include"types.h"
#include"usrinter.h"
#include"graphele.h"
  
class Element : public Object {

protected:
GraphElement *g; //pointer to it's graphic element

public:
virtual void draw(int,int); //for drawing the element                     
virtual typ is_a(void)=0;   //abstract for returning what it is     
void gset(GraphElement*);   //set the pointer to the graphic element     

};

#endif 
