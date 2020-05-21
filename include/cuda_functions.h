#ifndef __KERN_DUDA__H__
#define __KERN_DUDA__H__

#include <roswacocuda.h>
#include <cuda.h>

#define IK_VERSION 61
#define IKFAST_NO_MAIN true
#define IKFAST_HAS_LIBRARY

#ifdef IKFAST_NAMESPACE
  #undef IKFAST_NAMESPACE
  #define IKFAST_NAMESPACE tool_robot
  #include "ikfast.h"
  #undef IKFAST_NAMESPACE
#else
  #define IKFAST_NAMESPACE tool_robot
  #include "ikfast.h"
  #undef IKFAST_NAMESPACE
#endif

#ifdef IKFAST_NAMESPACE
  #undef IKFAST_NAMESPACE
  #define IKFAST_NAMESPACE positioner_robot
  #include "ikfast.h"
  #undef IKFAST_NAMESPACE
#else
  #define IKFAST_NAMESPACE positioner_robot
  #include "ikfast.h"
  #undef IKFAST_NAMESPACE
#endif


__global__ void AntHeuristic(void* heur,void* graph_ptr, bool* feasible, float beta,joint_boundary* robot2_speed_lim,Kine_const kine_param,
                             float* timevector, int* multiturn_axis);

__global__ void AntCycle(void* graph,int* solution,int* sol,float* maxfobjtot,void* heur,bool* feasible,ACO_params ACO,bool* coll_check,
                         Kine_const kine_param, int* multiturn_axis, int* logantsdev,float* logphdev,float* timevector, 
                         joint_boundary *joint_limits, float* real_joints_angle);

__global__ void AntCycle_Heur_online(void* graph,int* solution,int* sol,float* maxfobjtot,void* heur_onl,bool* feasible,ACO_params ACO,
                                    bool* coll_check,Kine_const kine_param, int* multiturn_axis, int* logantsdev,float* logphdev,
                                    float* timevector, joint_boundary *joint_limits,float* real_joint_angle, 
                                    joint_boundary *joint_limits_speed, float beta);

__global__ void WOACycle(joint_boundary* robot1_jnt_lim, joint_boundary* robot2_jnt_lim ,int w_iter,Eigen::Matrix4f* pathrel,float* timevector
                        ,Eigen::Matrix4f RobotsTrans,Eigen::Matrix4f gripper_trans,void* graph,void* joints_pos
                        ,bool* checkreach, Kine_const kine_param, Eigen::Matrix4f* inv_tool_trans,
                        Eigen::Matrix4f* Robot_gripper,int* max_conf_graph_dev ,tool_robot::IkReal* solIK,float* hal1,float* hal2,float* hal3);

__global__ void WOABestAndHeuristic(void *bestjointpos, void *joints_pos,int cyc, WOA_params WOA,float *WhaleScores, float *bestScore,int *Antpaths, 
                                    void *graph,void *best_path_angles, Kine_const kine_param, 
                                    joint_boundary* robot1_jnt_lim,int whale_check_coll);






#endif
