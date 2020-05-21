#include <roswacocuda.h>
#include <stdlib.h>
#include "string"
#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>

#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/ManagedMap.h>

std::string Robot1;
std::string Robot2;
std::string Environment;

int ncoll;

gpu_voxels::GpuVoxelsSharedPtr gvl;
std::string Robot_1_Voxellist;
std::string Robot_2_Voxellist;
std::map<std::string,float> myRobotJointValues, myRobotJointValues_2;
std::vector<std::string> Robot1JointNames;
std::vector<std::string> Robot2JointNames;

std::string Robot_3_Voxellist_name;
std::string Robot_4_Voxellist_name;
std::string Environment_Voxelmap;


std::string Robot3;
std::string Robot4;

std::vector<std::string> Robot3JointNames;
std::vector<std::string> Robot4JointNames;

  BitVectorVoxel bits_in_collision;


int main()
{
  float voxel_size=0.05;
  float map_size_x=5;
  float map_size_y=5;
  float map_size_z=3;
  int voxel_x=map_size_x/voxel_size;
  int voxel_y=map_size_y/voxel_size;
  int voxel_z=map_size_z/voxel_size;
  
  
  Robot1="Robot1";
  Robot2="Robot2";
  Environment="Environment";
  
  Robot1JointNames.resize(6);
  Robot2JointNames.resize(6);
  
  Robot1JointNames[0]="joint_1";
  Robot1JointNames[1]="joint_2";
  Robot1JointNames[2]="joint_3";//prendere dall'urdf
  Robot1JointNames[3]="joint_4";
  Robot1JointNames[4]="joint_5";
  Robot1JointNames[5]="joint_6";
  
  Robot2JointNames[0]="joint_1";
  Robot2JointNames[1]="joint_2";
  Robot2JointNames[2]="joint_3";
  Robot2JointNames[3]="joint_4";
  Robot2JointNames[4]="joint_5";
  Robot2JointNames[5]="joint_6";
  
  Robot_1_Voxellist = "Robot_1_Voxellist";
  Robot_2_Voxellist = "Robot_2_Voxellist";
  Environment_Voxelmap = "Environment_Voxelmap";

  gvl = GpuVoxels::getInstance();
  gvl->GpuVoxels::initialize(voxel_x, voxel_y, voxel_z, voxel_size); // map of x-y-z voxels each one of voxels_size dimension

  gvl->GpuVoxels::addMap(MT_BITVECTOR_VOXELLIST, Robot_1_Voxellist);
  gvl->GpuVoxels::addMap(MT_BITVECTOR_VOXELLIST, Robot_2_Voxellist);
  gvl->GpuVoxels::addMap(gpu_voxels::MT_BITVECTOR_VOXELMAP, Environment_Voxelmap);
  
  gvl->GpuVoxels::addRobot(Robot1, "/home/tartaglia/catkin_ws/src/itia_ros_wacocuda/Urdf_assembled/Evolaser/robot_gripper/robot_gripper_plate.urdf", true);
  gvl->GpuVoxels::addRobot(Robot2, "/home/tartaglia/catkin_ws/src/itia_ros_wacocuda/Urdf_assembled/Evolaser/robot_tool/robot_tool.urdf", true);
  gvl->GpuVoxels::addRobot(Environment, "/home/tartaglia/catkin_ws/src/itia_ros_wacocuda/Urdf_assembled/Evolaser/environment/environment.urdf", true);
  
  gvl->insertRobotIntoMap(Environment, Environment_Voxelmap, eBVM_OCCUPIED);

  
  
  myRobotJointValues[Robot1JointNames[0]] = 0.842048;   
  myRobotJointValues[Robot1JointNames[1]] = 0.714431;   
  myRobotJointValues[Robot1JointNames[2]] = -1.079710;
  myRobotJointValues[Robot1JointNames[3]] = 1.410780;  
  myRobotJointValues[Robot1JointNames[4]] = -0.864608;
  myRobotJointValues[Robot1JointNames[5]] = 0.835240;
  
  myRobotJointValues_2[Robot2JointNames[0]] = -0.708356 ;
  myRobotJointValues_2[Robot2JointNames[1]] = +0.819560 ;
  myRobotJointValues_2[Robot2JointNames[2]] = -0.157122 ;
  myRobotJointValues_2[Robot2JointNames[3]] = -1.595769 ;
  myRobotJointValues_2[Robot2JointNames[4]] = +0.500926 ;
  myRobotJointValues_2[Robot2JointNames[5]] = -0.889178 ;

  
  gvl->setRobotConfiguration(Robot1, myRobotJointValues);
  gvl->insertRobotIntoMap(Robot1, Robot_1_Voxellist, eBVM_OCCUPIED);
  
  gvl->setRobotConfiguration(Robot2, myRobotJointValues_2);
  gvl->insertRobotIntoMap(Robot2, Robot_2_Voxellist, eBVM_OCCUPIED);
  
  
  ncoll = gvl->getMap(Robot_2_Voxellist)->as<voxellist::BitVectorVoxelList>()->
          collideWithTypes(gvl->getMap(Environment_Voxelmap)->as<voxelmap::BitVectorVoxelMap>(), bits_in_collision);
  
  while(1)
  {
//     gvl->visualizeMap(Robot_1_Voxellist);
    gvl->visualizeMap(Robot_2_Voxellist);
//     gvl->visualizeMap(Environment_Voxelmap);
    
    std::cout  << "coll:\t"<< ncoll << "\n";
    usleep(1000000);
  }
}