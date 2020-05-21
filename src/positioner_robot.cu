#include <stdio.h>
#define IKFAST_NO_MAIN 
#define IKFAST_HAS_LIBRARY 

#ifdef IKFAST_NAMESPACE

  #undef IKFAST_NAMESPACE
  #define IKFAST_NAMESPACE positioner_robot
  
#else
  #define IKFAST_NAMESPACE positioner_robot
  
#endif

#include "ikfast.h"
#include <positioner.cu>
  
#undef IKFAST_NAMESPACE
