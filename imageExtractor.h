#ifndef __IMAGEEXTRACTOR_H__  
#define __IMAGEEXTRACTOR_H__  

struct metadata
{
    string eventType;
    float impactParameter;
    unsigned int eventID;
    unsigned int telNum;
    UShort_t MCprim;
    float MCe0;
    float MCxcore;
    float MCycore;
    float telx;
    float tely;
    float MCze;
    float MCaz;
    float MCxoff;
    float MCyoff;
    int pedrms;
};

#endif 
