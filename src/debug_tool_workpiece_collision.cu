#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/JointState.h>
#include <tf/transform_broadcaster.h>
 
#include <sstream>
#include <fstream>
#include <string>

#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/GpuVoxels.h>

#include <csv.h>
#include "iostream"

#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>

#include <gpu_voxels/ManagedMap.h>

#include <sstream>
#include <fstream>
#include <string>



std::string Robot1;
std::string Robot2;

gpu_voxels::GpuVoxelsSharedPtr gvl;
std::string Robot_1_Voxellist;
std::string Robot_2_Voxellist;
std::map<std::string,float> myRobotJointValues, myRobotJointValues_2;
std::vector<std::string> Robot1JointNames;
std::vector<std::string> Robot2JointNames;

std::string Robot_3_Voxellist_name;
std::string Robot_4_Voxellist_name;


std::string Robot3;
std::string Robot4;

std::vector<std::string> Robot3JointNames;
std::vector<std::string> Robot4JointNames;



int main(int argc, char *argv[])
{
  float tool_voxel_size=0.005;
  float tool_map_size_x=5;
  float tool_map_size_y=5;
  float tool_map_size_z=3;
  int tool_voxel_x=tool_map_size_x/tool_voxel_size;
  int tool_voxel_y=tool_map_size_y/tool_voxel_size;
  int tool_voxel_z=tool_map_size_z/tool_voxel_size;
  
  
  float voxel_size=0.05;
  float map_size_x=5;
  float map_size_y=5;
  float map_size_z=3;
  int voxel_x=map_size_x/voxel_size;
  int voxel_y=map_size_y/voxel_size;
  int voxel_z=map_size_z/voxel_size;
  
  
  Robot1 = "Gripper";                                                                          
  Robot2 = "Tool";                                                                             
  Robot1JointNames.resize(6);
  Robot2JointNames.resize(6);
  
  Robot1JointNames[0]="joint_1";
  Robot1JointNames[1]="joint_2";
  Robot1JointNames[2]="joint_3";
  Robot1JointNames[3]="joint_4";
  Robot1JointNames[4]="joint_5";
  Robot1JointNames[5]="joint_6";
  
  Robot2JointNames[0]="joint_1";
  Robot2JointNames[1]="joint_2";
  Robot2JointNames[2]="joint_3";
  Robot2JointNames[3]="joint_4";
  Robot2JointNames[4]="joint_5";
  Robot2JointNames[5]="joint_6";
  
  std::string tool = "tool";
  std::string gripper_workpiece = "gripper_workpiece";
  
  gvl = GpuVoxels::getInstance();
  
  gvl->GpuVoxels::initialize(tool_voxel_x, tool_voxel_y, tool_voxel_z, tool_voxel_size); // map of x-y-z voxels each one of voxels_size dimension

  gvl->GpuVoxels::addMap(MT_BITVECTOR_VOXELLIST, gripper_workpiece);
  gvl->GpuVoxels::addMap(MT_BITVECTOR_VOXELLIST, tool);  
  
  gvl->GpuVoxels::addRobot(Robot1, "/home/tartaglia/catkin_ws/src/itia_ros_wacocuda/Urdf_assembled/Evolaser/gripper_workpiece/gripper_plate_joints.urdf", true);
  gvl->GpuVoxels::addRobot(Robot2, "/home/tartaglia/catkin_ws/src/itia_ros_wacocuda/Urdf_assembled/Evolaser/tool/tool_joints.urdf", true);

  float robjoint[6*13];
  
  std::ifstream infile;
  
  infile.open("/media/DATA/WACO/Articolo_2018/Risultati_numerici/1_50_256_200/Robot.txt");

  if (infile.fail())
  {
    std::cout << "error"<< "\n";
  }
  else
  {
    float value=0;
    int i=0;
    
    while (infile >> value )
    {
      robjoint[i]=value;
      i++;
    }
  }
  
  float joint_1_csv,joint_2_csv,joint_3_csv,joint_4_csv,joint_5_csv,joint_6_csv;
  float joint_1,joint_2,joint_3,joint_4,joint_5,joint_6;
  
  io::CSVReader<6> gripper_pose("/media/DATA/WACO/Articolo_2018/Risultati_numerici/1_50_256_200/positioner.txt");
  gripper_pose.set_header("joint_1_csv","joint_2_csv","joint_3_csv","joint_4_csv","joint_5_csv","joint_6_csv");
    while(gripper_pose.read_row(joint_1_csv,joint_2_csv,joint_3_csv,joint_4_csv,joint_5_csv,joint_6_csv))
  {
    
    joint_1=joint_1_csv;
    joint_2=joint_2_csv;
    joint_3=joint_3_csv;
    joint_4=joint_4_csv;
    joint_5=joint_5_csv;
    joint_6=joint_6_csv;
    
    std::cout << joint_1 << "\t" << joint_2 << "\t" << joint_3 << "\t" << joint_4 << "\t" << joint_5 << "\t" << joint_6 << "\n";
  }
  

  int ncoll, ncoll_2;
  BitVectorVoxel bits_in_collision;
  
  ncoll=-1;
 

  int next_point=0;
int index=0;

  while(1)
  {
    
    if (next_point=1)
    {
      next_point=0;
      index ++;
      gvl->clearMap(tool);
      gvl->clearMap(gripper_workpiece);
      if (index>12)
      {
        
        index=0;
      }
    }
    
    myRobotJointValues_2[Robot2JointNames[0]] = robjoint[index*6]; 
    myRobotJointValues_2[Robot2JointNames[1]] = robjoint[index*6+1];
    myRobotJointValues_2[Robot2JointNames[2]] = robjoint[index*6+2];
    myRobotJointValues_2[Robot2JointNames[3]] = robjoint[index*6+3];
    myRobotJointValues_2[Robot2JointNames[4]] = robjoint[index*6+4];
    myRobotJointValues_2[Robot2JointNames[5]] = robjoint[index*6+5];
                      
    
    myRobotJointValues[Robot1JointNames[0]] = joint_1;
    myRobotJointValues[Robot1JointNames[1]] = joint_2;
    myRobotJointValues[Robot1JointNames[2]] = joint_3;
    myRobotJointValues[Robot1JointNames[3]] = joint_4;
    myRobotJointValues[Robot1JointNames[4]] = joint_5;
    myRobotJointValues[Robot1JointNames[5]] = joint_6;
                      
    gvl->setRobotConfiguration(Robot1, myRobotJointValues);
    gvl->insertRobotIntoMap(Robot1, gripper_workpiece, eBVM_OCCUPIED);
    
    gvl->setRobotConfiguration(Robot2, myRobotJointValues_2);
    gvl->insertRobotIntoMap(Robot2, tool, eBVM_OCCUPIED);
    
    gvl->visualizeMap(gripper_workpiece);
    gvl->visualizeMap(tool);
    
    ncoll = gvl->getMap(gripper_workpiece)->as<voxellist::BitVectorVoxelList>()->
        collideWithTypes(gvl->getMap(tool)->as<voxellist::BitVectorVoxelList>(),bits_in_collision);

    usleep(1000000);    
        
    std::cout <<"point:\t"<< index << "\tncoll:\t"<< ncoll << "\n";
    std::cin >> next_point;
    
     
  }
}

  
