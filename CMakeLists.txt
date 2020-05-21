cmake_minimum_required(VERSION 3.5.1)
set (CMAKE_CXX_STANDARD 11)
project(itia_ros_wacocuda CXX)


if(NOT DEFINED GPU_VOXEL_CMAKE_PATH)
  message( FATAL_ERROR "you have to provide a GPU_VOXEL_CMAKE_PATH." )
endif()

if(NOT DEFINED ICL_CORE_CMAKE_PATH)
  message( FATAL_ERROR "you have to provide a ICL_CORE_CMAKE_PATH." )
endif()

set( GPU_VOXEL_CMAKE_PATH "" CACHE STRING "" )
set( ICL_CORE_CMAKE_PATH  "" CACHE STRING "" )

MESSAGE(WARNING "GPU_VOXEL_CMAKE_PATH=${GPU_VOXEL_CMAKE_PATH}")
MESSAGE(WARNING "ICL_CORE_CMAKE_PATH=${ICL_CORE_CMAKE_PATH}")


find_package(CUDA 10 REQUIRED)

if (CUDA_FOUND)
  #Get CUDA compute capability
  set(OUTPUTFILE ${CMAKE_CURRENT_BINARY_DIR}/itia_ros_wacocuda) # No suffix required
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/src/check_cuda.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE ARCH)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message(WARNING "CUDA Architecture: ${ARCH}")
    message(WARNING "CUDA Version: ${CUDA_VERSION_STRING}")
    message(WARNING "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(WARNING "CUDA Libararies: ${CUDA_LIBRARIES}")
    message(WARNING "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")

  else()
    message(WARNING ${ARCH})
  endif()
endif()




FIND_PACKAGE(icl_core REQUIRED HINTS ${ICL_CORE_CMAKE_PATH} NO_DEFAULT_PATH)
FIND_PACKAGE(gpu_voxels REQUIRED HINTS ${GPU_VOXEL_CMAKE_PATH} NO_DEFAULT_PATH )
FIND_PACKAGE(Boost COMPONENTS system REQUIRED)
FIND_PACKAGE(orocos_kdl REQUIRED)

#  add_definitions(-DDEBUG_HEUR_GRAPH)
#  add_definitions(-DDEBUG_HEUR)
#  add_definitions(-DDEBUG_HEUR_POS)
#  add_definitions(-DDEBUG_HEUR_ANTS)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    ${ARCH}   
    -std=c++11 
    -cudart=static 
    -rdc=true 
    -shared 
    -Xptxas -O2
    -Xcompiler -O2 
    -Xcompiler -fPIC 
    -maxrregcount 64 
    -use_fast_math 
    -w 
    --expt-relaxed-constexpr
)

set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_FLAGS})
add_definitions(${CUDA_NVCC_FLAGS})

message(WARNING "CUDA FLAGS  : ${CUDA_NVCC_FLAGS}")

set(CUDA_VERBOSE_BUILD ON)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(BUILD_SHARED_LIBS OFF)
set(CUDA_VERBOSE_BUILD ON)
# set(CUDA_PROPAGATE_HOST_FLAGS ON)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  actionlib_msgs 
  actionlib
  moveit_ros_planning_interface
  moveit_visual_tools
  std_msgs
  sensor_msgs
)

catkin_package(
   INCLUDE_DIRS include
   CATKIN_DEPENDS roscpp std_msgs sensor_msgs message_runtime
   DEPENDS 
   actionlib_msgs 
   actionlib  
)

add_definitions(-D__CUDA__ -D__CUDACC__ 
  ${icl_core_DEFINITIONS}
  ${gpu_voxels_DEFINITIONS}
  )

include_directories(
  ${catkin_INCLUDE_DIRS}
#   ${CMAKE_CURRENT_SOURCE_DIR}
  ${gpu_voxels_INCLUDE_DIRS}
  ${orocos_kdl_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  include
)

# cuda_add_library( tool_robot SHARED src/tool_robot.cu  OPTIONS -g -G)
# cuda_add_library( positioner_robot SHARED src/positioner_robot.cu OPTIONS -g -G)
# cuda_add_library( cuda_functions SHARED src/cuda_functions.cu src/tool_robot.cu src/positioner_robot.cu OPTIONS -g -G)

# 
# target_link_libraries(cuda_functions 
#                       tool_robot
#                       positioner_robot
#                       ${CUDA_LIBRARIES}
#                       )



cuda_add_executable(  itia_ros_wacocuda
                      src/cuda_functions.cu
                      src/roswacocuda.cu
                      src/positioner_robot.cu
                      src/tool_robot.cu
                      src/save_data.cu
                      src/collisions.cu
)

message(WARNING "CUDA_cudadevrt_LIBRARY : ${CUDA_cudadevrt_LIBRARY}")

target_link_libraries( itia_ros_wacocuda 
  ${CUDA_cudadevrt_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${icl_core_LIBRARIES}
  ${gpu_voxels_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${catkin_LIBRARIES}
)
# cuda_add_executable( preview_voxelized src/preview_voxelized.cu)
# 
# target_link_libraries( preview_voxelized
#   ${Boost_SYSTEM_LIBRARY}
#   ${gpu_voxels_LIBRARIES}
#   ${icl_core_LIBRARIES}
#   ${CUDA_LIBRARIES}
#   ${catkin_LIBRARIES}
# )
# 
cuda_add_executable( show_trajectory src/show_trajectory.cu)

target_link_libraries( show_trajectory
  ${Boost_SYSTEM_LIBRARY}
  ${gpu_voxels_LIBRARIES}
  ${icl_core_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${catkin_LIBRARIES}
)  
#   cuda_add_executable( debug_coll_tool_piece src/debug_tool_workpiece_collision.cu)
# 
# target_link_libraries( debug_coll_tool_piece
#   ${Boost_SYSTEM_LIBRARY}
#   ${gpu_voxels_LIBRARIES}
#   ${icl_core_LIBRARIES}
#   ${CUDA_LIBRARIES}
#   ${catkin_LIBRARIES}
#   )
#   
#   cuda_add_executable( debug_coll_posit_env src/debug_coll_posit_env.cu)
# 
# target_link_libraries( debug_coll_posit_env
#   ${Boost_SYSTEM_LIBRARY}
#   ${gpu_voxels_LIBRARIES}
#   ${icl_core_LIBRARIES}
#   ${CUDA_LIBRARIES}
#   ${catkin_LIBRARIES}
#   
# )
# 
  cuda_add_executable( positioner_debug src/positioner_debug.cu src/positioner_robot.cu )

target_link_libraries( positioner_debug
  ${Boost_SYSTEM_LIBRARY}
  ${gpu_voxels_LIBRARIES}
  ${icl_core_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${catkin_LIBRARIES}
  
)
