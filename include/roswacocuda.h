#ifndef __WACODUDA__H__
#define __WACODUDA__H__

#ifndef EIGEN_DEFAULT_DENSE_INDEX_TYPE
  #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#else

#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#endif

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <assert.h>
#include <unistd.h>
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/GpuVoxels.h>
#include <iostream>
#include "time.h"
#include "eigen3/Eigen/LU"
#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits>
#include <fstream>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <thrust/complex.h>
#include <unistd.h>
#include <ctime>
#include <cstdlib>
#include <stdint.h>
#include <cstdio>
#include <sys/mman.h>
#include <math_constants.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <urdf/model.h>
#include <moveit_msgs/ExecuteTrajectoryAction.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <gpu_voxels/GpuVoxels.h>
#include <csv.h>
#include <cuda_utilities.h>
#include <ros/package.h>
#define PI_F 3.141592654f

static bool checkSolNotEmpty(bool* vec,int length)
{
  for (int ind=0;ind<length;ind++) if(vec[ind]) return true;
  return false;
}

// struct joints 
// {
//     float jointsval[JOINTS];
//     bool ch = false;
//     float ph = 0.0;
// };

struct Kine_const{
  int JOINTS_POS;
  int JOINTS_TOOL;
  int TOT_SOL_IK;
  int PATH_PNT;
  float UPPER_RANGE;
  float LOWER_RANGE;
  float STEP;
  int Z_AXIS_DISCR;
  int MAX_RID_CONF;
  int MAX_RID_CONF_REAL;
  float MAX_OPEN_CONE_ANGLE;
  float STEP_OPEN_CONE_ANGLE;
  float STEP_ROT_CONE;
  int N_ROT_CONE;
  int RID_CONE;
  int N_OPEN_CONE;
  int RID_CONF;
};


struct joint_boundary{
  float upper;
  float lower;
  float middle;
};

struct boundaries{
  const int        n_joints; 
  joint_boundary*  pos_joint;
  joint_boundary*  speed_joint;
  int*             multiturn_bool;
  
  
  boundaries(const int njoints,int* multiturn_bool_) : n_joints( njoints) 
  {
    pos_joint      = n_joints > 0 ? new joint_boundary[n_joints] : NULL;
    speed_joint    = n_joints > 0 ? new joint_boundary[n_joints] : NULL;
    multiturn_bool = n_joints > 0 ? new int[n_joints]            : NULL;
    
    if (n_joints>0)
    {
      for (int jnt=0; jnt<n_joints; jnt++)
      {
        pos_joint->middle   = 0.5*(pos_joint[jnt].upper+pos_joint[jnt].middle);
        speed_joint->middle = 0.5*(speed_joint[jnt].upper+speed_joint[jnt].middle);
        multiturn_bool[jnt] = multiturn_bool_[jnt];
      }
    }
  }
  
  boundaries(const int njoints) : n_joints( njoints) 
  {
    pos_joint      = n_joints > 0 ? new joint_boundary[n_joints] : NULL;
    speed_joint    = n_joints > 0 ? new joint_boundary[n_joints] : NULL;
    
    if (n_joints>0)
    {
      for (int jnt=0; jnt<n_joints; jnt++)
      {
        pos_joint->middle=0.5*(pos_joint[jnt].upper+pos_joint[jnt].middle);
        speed_joint->middle=0.5*(speed_joint[jnt].upper+speed_joint[jnt].middle);
      }
    }
  }
  
  size_t size( )
  {
    return sizeof(int) + 2 * sizeof( joint_boundary ) * n_joints + n_joints * sizeof(int);
  }
};

struct trajectory{
  int n_points;
  float* timevector;
  Eigen::Matrix4f* points;

  trajectory(std::string trajectory_type,double radius, int n_pnt, double linear_speed)
  {
    n_points=n_pnt;
    
    timevector = n_points > 1 ? (float*)malloc((n_points-1)*sizeof(float)) : NULL;
    points = (Eigen::Matrix4f*)malloc(n_points*sizeof(Eigen::Matrix4f));
    

    if (trajectory_type=="cerchio")
    {
    
      float delta_time= ((1.3*PI_F*radius)/(n_points-1))/linear_speed;
      
      for(int pnt=0; pnt<n_points; pnt++)
      {
        points[pnt](0,0)=1; points[pnt](0,1)=0; points[pnt](0,2)=0; points[pnt](0,3)=(radius*cos(2*PI_F*((double)pnt/(n_points-1)))-radius);
        points[pnt](1,0)=0; points[pnt](1,1)=1; points[pnt](1,2)=0; points[pnt](1,3)=(radius*sin(2*PI_F*((double)pnt/(n_points-1))));
        points[pnt](2,0)=0; points[pnt](2,1)=0; points[pnt](2,2)=1; points[pnt](2,3)=0;
        points[pnt](3,0)=0; points[pnt](3,1)=0; points[pnt](3,2)=0; points[pnt](3,3)=1;
        
        timevector[pnt]=delta_time;
      }
    }
    else if (trajectory_type=="rifilo")
    {
    
      double delta_time= ((1.3*PI_F*radius)/(n_points-1))/linear_speed;
      
      for(int pnt=0; pnt<n_points; pnt++)
      {
        points[pnt](0,0)=cos(2*PI_F*((double)pnt/n_points)); points[pnt](0,1)=-sin(2*PI_F*((double)pnt/(n_points-1))); points[pnt](0,2)=0; points[pnt](0,3)=(radius*cos(2*PI_F*((double)pnt/(n_points-1)))-radius);
        points[pnt](1,0)=sin(2*PI_F*((double)pnt/n_points)); points[pnt](1,1)= cos(2*PI_F*((double)pnt/(n_points-1))); points[pnt](1,2)=0; points[pnt](1,3)=(radius*sin(2*PI_F*((double)pnt/(n_points-1))));
        points[pnt](2,0)=0                                 ; points[pnt](2,1)=0                                      ; points[pnt](2,2)=1; points[pnt](2,3)=0;
        points[pnt](3,0)=0                                 ; points[pnt](3,1)=0                                      ; points[pnt](3,2)=0; points[pnt](3,3)=1;
          
        timevector[pnt]=delta_time;
      }
    }
    else if (trajectory_type=="tubo_storto")
    {
      Eigen::Matrix4f T;
//       T(0,0)=0.3420201433257;   T(0,1)=0.0;    T(0,2)=0.9396926207859;  T(0,3)=0.1789065;
//       T(1,0)=0.9396926207859;   T(1,1)=0.0;    T(1,2)=-0.3420201433257; T(1,3)=0.0894261;
//       T(2,0)=0.0            ;   T(2,1)=1.0;    T(2,2)=0.0;              T(2,3)=0.0;
//       T(3,0)=0.0;               T(3,1)=0.0;    T(3,2)=0.0;              T(3,3)=1.0;
      
      T(0,0)=-0.9396926207859;   T(0,1)=-0.3420201433257;   T(0,2)=0.0;              T(0,3)=-0.1268683;
      T(1,0)=0.3420201433257;   T(1,1)=-0.9396926207859;   T(1,2)=0.0;              T(1,3)=0.1512;
      T(2,0)=0.0            ;   T(2,1)=0.0;                 T(2,2)=1.0;              T(2,3)=0.150;
      T(3,0)=0.0;               T(3,1)=0.0;                 T(3,2)=0.0;              T(3,3)=1.0;      
      
      float alfa=0;
      double delta_time= (((2*PI_F*radius)/n_points))/linear_speed;
      Eigen::Matrix4f rifilo;
      
      for(int pnt=0; pnt<n_points; pnt++)
      {
        alfa=pnt*(2*PI_F/n_points);
        
        rifilo(0,0)=sin(alfa);   rifilo(0,1)=0;  rifilo(0,2)=-cos(alfa);   rifilo(0,3)=radius*cos(alfa);
        rifilo(1,0)=-cos(alfa);  rifilo(1,1)=0;  rifilo(1,2)=-sin(alfa);   rifilo(1,3)=radius*sin(alfa);
        rifilo(2,0)=0;           rifilo(2,1)=1;  rifilo(2,2)=0;            rifilo(2,3)=0;
        rifilo(3,0)=0;           rifilo(3,1)=0;  rifilo(3,2)=0;            rifilo(3,3)=1;
        
        points[pnt]=T*rifilo;
        
        timevector[pnt]=delta_time;
      }
    }
  }
  
  trajectory(std::string path)
  {
    Eigen::Quaternionf q;
    Eigen::Matrix3f R ;
    std::vector<Eigen::Matrix4f> TraspVect;
    std::vector<float> TmpTime;
    Eigen::Matrix4f Trasp;
    
    
    io::CSVReader<8> TrajFile(path);
    TrajFile.set_header("x0", "y0", "z0","w", "x", "y", "z", "t");
    float x0,y0,z0,x,y,z,w,t;
    while(TrajFile.read_row(x0,y0,z0,w,x,y,z,t))
    {
      q.x() = x;
      q.y() = y;
      q.z() = z;
      q.w() = w;
      R = q.normalized().toRotationMatrix();
      
      Trasp(0,0)=R(0,0);Trasp(0,1)=R(0,1);Trasp(0,2)=R(0,2);Trasp(0,3)=x0;
      Trasp(1,0)=R(1,0);Trasp(1,1)=R(1,1);Trasp(1,2)=R(1,2);Trasp(1,3)=y0;
      Trasp(2,0)=R(2,0);Trasp(2,1)=R(2,1);Trasp(2,2)=R(2,2);Trasp(2,3)=z0;
      Trasp(3,0)=0;     Trasp(3,1)=0;     Trasp(3,2)=0;     Trasp(3,3)=1;
      TraspVect.push_back(Trasp);
      TmpTime.push_back(t);
      
    }
    n_points=TraspVect.size();
    printf("Points in the trajectory : %d \n",n_points);
    timevector = n_points > 1 ? (float*)malloc((n_points-1)*sizeof(float)) : NULL;
    points = n_points > 1 ? (Eigen::Matrix4f*)malloc(n_points*sizeof(Eigen::Matrix4f)) : NULL;
    
    for(int pnt=0; pnt<n_points; pnt++)
    {
      points[pnt]=TraspVect[pnt];
      if (pnt>0)
      {
        timevector[pnt-1]=TmpTime[pnt]-TmpTime[pnt-1];
      }
    }
  }

  int get_points_num()
  {
    return n_points;
  }
};

struct collision_data{
  std::string Robot1;
  std::string Robot2;
  std::string Environment;
  std::string Robot_1_Voxellist;
  std::string Robot_2_Voxellist;
  std::string Environment_Voxelmap;
  std::vector<std::string> Robot1JointNames;
  std::vector<std::string> Robot2JointNames;  
  std::map<std::string,float> myRobotJointValues;
  std::map<std::string,float> myRobotJointValues2;
  int ncoll;
};

struct WOA_params{
  int whales;
  int cycles;
  float A_lim;
  size_t shr_bytes_heuWOA;
};

struct ACO_params{
  int ants;
  int cycles;
  int nodes;
  float PHevap;
  float beta;
  size_t shr_bytes_ACO;
};

struct optim_param{
  WOA_params WOA;
  ACO_params ACO;
};



////////////// CLASS DEFINITION

class WACOCuda
{

  int* soldev;
  int* solhost;
  int* host_ants_path;
  int* dev_ants_path; 
  int* logantsdev;
  int* logantshost;
  int* dev_multiturn_axis;
  int* dev_IK_pntcoutercheck;
  int* max_conf_graph_dev;
  int* max_conf_graph_host;
  
  int check_whale_coll;
  int counter_better_paths;
  
  float* deviceBestscore;
  float* dev_maxFObjWhales;
  float* host_maxFObjWhales;
  float* dev_timevector; 
  float* logphdev;
  float* logphhost;
  float* dev_real_joint_angle;

  float* hal1_dev;
  float* hal2_dev;
  float* hal3_dev;
  
  void *devicejointbest;
  void *devsoltraj;
  void *jointshost;
  void *jointsdev;
  void *devgraph;
  void *hostgraph;
  void *hostgraphempty;
  void *graph_checked;
  void *hostjointpos;
  void *devjointpos;
  void *heuristichost_;
  void *heuristicdev_;
  void *dev_heur_onl;
  
  const boundaries& robot_pos_limits;
  const boundaries& robot_tool_limits;
  
  bool *devcheckreach;
  bool *hostcheckreach;
  bool *precheckcoll;
  bool *already_checked_coll;
  bool *dev_already_checked_coll;
  
  bool *toolrob_env_check;
  bool *toolrob_gripp_check;
  bool *toolrob_posit_check;
  
  bool no_collisions;
  bool heuristic_online;
  
  
  cudaError_t check_mem_;
  
  size_t shrbytes;
  
  Eigen::Matrix4f RobotsTrans;//Relative transform from robot 1 to robot 2
  Eigen::Matrix4f gripper_trans;  //Relative transform between positioner TCP and gripper TCP
  Eigen::Matrix4f inv_tool_trans;   //Relative transform from tool TCP to robot

  Eigen::Matrix4f* devpathsol;
  Eigen::Matrix4f* dev_absolute_path;
  Eigen::Matrix4f* dev_pathrel;
  Eigen::Matrix4f* dev_pos_tcp;
  Eigen::Matrix4f* host_pos_tcp;
  Eigen::Matrix4f* dev_Robot_Gripper_tcp;
  Eigen::Matrix4f* dev_inv_tool_trans;
  
  collision_data collisiondata;
  collision_data tool_gripper;
  optim_param optim_param;
  Kine_const Kine_param;
  joint_boundary* dev_robot_pos_lim_joint;
  joint_boundary* dev_robot_tool_lim_joint;
  joint_boundary* dev_robot_pos_speed_joint;
  joint_boundary* dev_robot_tool_speed_joint;
  
  gpu_voxels::GpuVoxelsSharedPtr gvl;
  voxellist::BitVectorVoxelList* tool;
  voxellist::BitVectorVoxelList* gripper_workpiece;
  
  std::string strdate;
  std::ostringstream oss;
  
  const std::vector<std::string> _PosJointsNames;
  const std::vector<std::string> _RobJointsNames;
  std::vector<std::string> _urdf_paths;
  
  double* solIK                   ; 
  
  public:
    void *hostjointbest;
    void *hostsoltraj;
    float *hostBestscore;
    Eigen::Matrix4f* hostpathsol;
    float* fobj_history;
    
    WACOCuda(int nwhales,int ncyc,const boundaries& limits, const boundaries& laserlimitsex,float factor,trajectory* pathex, 
             Eigen::Matrix4f RobotsTransex, Eigen::Matrix4f GrabTransex, int ants, int ant_cycles,Kine_const* KINE_param,
             Eigen::Matrix4f RobotTCP_tool,float beta,std::vector<std::string> PosJointsNames,std::vector<std::string> RobJointsNames,std::vector<std::string> urdf_paths);

    void RunWACO();
    void Copytohost();
    void AllocMem(int n_whales,int PATH_PNT,int MAX_RID_CONF_REAL);
    
    ~WACOCuda()
    {
//     TODO
      free(toolrob_env_check);free(toolrob_gripp_check);free(toolrob_posit_check);
      free(hostgraph);free(hostgraphempty);cudaFree(devgraph);
      cudaFree(solIK);
      free(max_conf_graph_host);cudaFree(max_conf_graph_dev);
      free(hostsoltraj);cudaFree(devsoltraj);
      free(hostjointbest);cudaFree(devicejointbest);
      free(host_maxFObjWhales);cudaFree(dev_maxFObjWhales);
      free(hostBestscore);cudaFree(deviceBestscore);
      free(hostcheckreach);cudaFree(devcheckreach);
      free(hostjointpos);cudaFree(devjointpos);
      free(host_ants_path);cudaFree(dev_ants_path);
      free(already_checked_coll);cudaFree(dev_already_checked_coll);
      free(hostpathsol);cudaFree(devpathsol);
      cudaFreeHost(host_pos_tcp);cudaFree(dev_pos_tcp);cudaFree(soldev);
      cudaFree(dev_robot_pos_lim_joint);cudaFree(dev_robot_tool_lim_joint);
      cudaFree(dev_robot_pos_speed_joint);cudaFree(dev_robot_tool_speed_joint);
      cudaFree(dev_pathrel);cudaFree(dev_timevector);
      cudaFree(dev_pos_tcp);cudaFree(dev_Robot_Gripper_tcp);cudaFree(dev_real_joint_angle);
      cudaFree(dev_inv_tool_trans);
      
      cudaFree(devsoltraj);
      cudaFree(jointsdev);
      cudaFree(dev_ants_path);
    };
    
  private:
    
    void InitializeCollisions();
    void RunWhaleCycle();
    void TestTrajRos(int argc2,char** argv2);
    void CollisionCheck();
    void LogAnts(std::string path,int wacyc);
    void CollisionCheck_afterACO();
    void ClearDevVariables();
    
    inline void Update_gripper_gpu_voxels(Multi_Positioner_Helper Positioner, int whale);
    inline void Update_tool_gpu_voxels(Multi_Graph_Helper Graph, int whale ,int pnt, int config);
    inline void Update_tool_gripper_gpu_voxels(Multi_Graph_Helper Graph, int whale ,int pnt, int config,std::string gripper_or_tool);
    inline void Update_positioner_gpu_voxels(Multi_Positioner_Helper Positioner, int whale);
    inline void Update_robot_gpu_voxels(Multi_Graph_Helper Graph, int whale, int pnt,int rid_config);

    bool Collision_Positioner_Environment(Multi_Positioner_Helper Positioner, int i_whale, Multi_Graph_Helper Graph);
    bool Collision_Gripper_Tool(Multi_Positioner_Helper Positioner, Multi_Graph_Helper Graph,int i_whale,bool* feasible,bool* check_coll_all_points);
    bool Collision_Toolrobot_Environment(Multi_Graph_Helper Graph,int i_whale, bool* feasible,bool* check_coll_all_points);
    bool Collision_Positioner_Toolrobot(Multi_Positioner_Helper Positioner, Multi_Graph_Helper Graph, int i_whale,bool* feasible,bool* check_coll_all_points);
    
    void WOABest_precheck();
    
    inline void Copy_after_AntHeu();
    inline void Copy_after_ACO();
    inline void Copy_after_CollisionCheck();
};

#endif
