# itia_WACOCuda

WACO algorithm implementation with CUDA c++   
CUDA-9.1,EIGEN,SISL(cuda version) are dependencies, the last included in the package. 
The WACOCuda class implements the WACO algorithm.    

Main already includes a working example.    
Algorithm parameters can be set in the WACOCuda.h file, using provided #defines.     
To change the objective function used by the ant algorihtm, modify the AntCycle function(wacocuda.cu file,around line 590) and uncomment the selected objective function, remember to comment all but one objective function.    
The default selected function minimizes velocity inversions.   
The code is still full of commented out parts, helpful to have feedback on parameters settings during debug.    

The package includes a special version of ikfast.py able to generate code to be included in CUDA projects.  
To generate the code for a specific robot, use the ikfast.py normally, then use the template.cu file found in the openrvaecuda folder to include the code in the main src file.  
The file ikfast.h has alsop been changed in order to work with cuda.  
Remeber to change the template.cu file to match your file name and namespace you want, and to include it in your project.  

build like a normal ROS package.

rosrun itia_ros_wacocuda (whalenumber) (whalecycles) (whalefactor)     // 0<whalenumber<1000  0<whalecycles  0<whalefactor<2  
 