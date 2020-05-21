#include <stdio.h>
#define IKFAST_NO_MAIN 
#define IKFAST_HAS_LIBRARY 

#ifdef IKFAST_NAMESPACE
  
  #undef IKFAST_NAMESPACE
  #define IKFAST_NAMESPACE tool_robot
  
#else
  #define IKFAST_NAMESPACE tool_robot
  
#endif

#include "ikfast.h"
#include "tool.cu"

#undef IKFAST_NAMESPACE
