#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/GpuVoxels.h>
#include <roswacocuda.h>

#include <signal.h>
#include "time.h"
#include "iostream"


 collision_data* collisiondata;
 gpu_voxels::GpuVoxelsSharedPtr gvl;

void ctrlchandler(int)
{
  gvl.reset();
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  gvl.reset();
  exit(EXIT_SUCCESS);
}


int main(int argc, char *argv[])
{
  
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);
  
  collisiondata = new collision_data() ; 
  
/*  float voxel_size=atoi(argv[0]);	//stessa unità di misura dell'URDF
  float map_size_x=atoi(argv[1]);
  float map_size_y=atoi(argv[2]);
  float map_size_z=atoi(argv[3])*/;
  
  float voxel_size=0.05;
  float map_size_x=5;
  float map_size_y=5;
  float map_size_z=3;
  
  
  int voxel_x=map_size_x/voxel_size;
  int voxel_y=map_size_y/voxel_size;
  int voxel_z=map_size_z/voxel_size;
  
  clock_t time1,time2;
  
  collisiondata->Robot1="Robot1";
  collisiondata->Robot2="Robot2";
  collisiondata->Environment_Voxelmap = "Environment_Voxelmap";
  
  collisiondata->Robot1JointNames.resize(6);
  collisiondata->Robot2JointNames.resize(6);
  
  collisiondata->Robot1JointNames[0]="joint_1";
  collisiondata->Robot1JointNames[1]="joint_2";
  collisiondata->Robot1JointNames[2]="joint_3";
  collisiondata->Robot1JointNames[3]="joint_4";
  collisiondata->Robot1JointNames[4]="joint_5";
  collisiondata->Robot1JointNames[5]="joint_6";
  
  collisiondata->Robot2JointNames[0]="joint_1";
  collisiondata->Robot2JointNames[1]="joint_2";
  collisiondata->Robot2JointNames[2]="joint_3";
  collisiondata->Robot2JointNames[3]="joint_4";
  collisiondata->Robot2JointNames[4]="joint_5";
  collisiondata->Robot2JointNames[5]="joint_6";
  
  collisiondata->Robot_1_Voxellist = "Robot_1_Voxellist";
  collisiondata->Robot_2_Voxellist = "Robot_2_Voxellist";

  gvl = GpuVoxels::getInstance();
  gvl->GpuVoxels::initialize(voxel_x, voxel_y, voxel_z, voxel_size); // map of x-y-z voxels each one of voxels_size dimension

  gvl->GpuVoxels::addMap(MT_BITVECTOR_VOXELLIST, collisiondata->Robot_1_Voxellist);
  gvl->GpuVoxels::addMap(MT_BITVECTOR_VOXELLIST, collisiondata->Robot_2_Voxellist);
  gvl->GpuVoxels::addMap(gpu_voxels::MT_BITVECTOR_VOXELMAP, collisiondata->Environment_Voxelmap);
  
  
  gvl->GpuVoxels::addRobot(collisiondata->Robot1, "/Urdf_assembled/Evolaser/robot_gripper/robot_gripper.urdf", true);
  gvl->GpuVoxels::addRobot(collisiondata->Robot2, "/Urdf_assembled/Evolaser/robot_tool/robot_tool.urdf", true);
  gvl->GpuVoxels::addRobot(collisiondata->Environment, "/Urdf_assembled/Evolaser/environment/environment.urdf", true);
  
  gvl->insertRobotIntoMap(collisiondata->Environment, collisiondata->Environment_Voxelmap, eBVM_OCCUPIED);
  
  collisiondata->myRobotJointValues[collisiondata->Robot1JointNames[0]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot1JointNames[1]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot1JointNames[2]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot1JointNames[3]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot1JointNames[4]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot1JointNames[5]]=0;
  
  collisiondata->myRobotJointValues[collisiondata->Robot2JointNames[0]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot2JointNames[1]]=-1;
  collisiondata->myRobotJointValues[collisiondata->Robot2JointNames[2]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot2JointNames[3]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot2JointNames[4]]=0;
  collisiondata->myRobotJointValues[collisiondata->Robot2JointNames[5]]=0;
  
  BitVectorVoxel bits_in_collision;
  
  int counter=0;
  double avg_coll_time=0;
  double avg_inserction_time=0;
  
  std::cout << "Start" << std::endl;
  
  gpu_voxels::Vector3i offset;
  
  offset.x=0.5*map_size_x/voxel_size;
  offset.y=0.5*map_size_y/voxel_size;
  offset.z=0.5*map_size_z/voxel_size;
  
  float treshold=1.0;
  
  while (true)
  {
  
    time2=clock();
    
    gvl->clearMap(collisiondata->Robot_1_Voxellist);
    gvl->clearMap(collisiondata->Robot_2_Voxellist);
    
    gvl->setRobotConfiguration(collisiondata->Robot1, collisiondata->myRobotJointValues);
    gvl->insertRobotIntoMap(collisiondata->Robot1, collisiondata->Robot_1_Voxellist, eBVM_OCCUPIED);
    
    gvl->setRobotConfiguration(collisiondata->Robot2, collisiondata->myRobotJointValues);
    gvl->insertRobotIntoMap(collisiondata->Robot2, collisiondata->Robot_2_Voxellist, eBVM_OCCUPIED);
    
    avg_inserction_time = avg_inserction_time + (double)(clock()-time2)/CLOCKS_PER_SEC;
    
    gvl->visualizeMap(collisiondata->Robot_1_Voxellist);
    gvl->visualizeMap(collisiondata->Robot_2_Voxellist);
    
    
    time1=clock();
    
//     collisiondata->ncoll = collisiondata->gvl->getMap(collisiondata->Robot_1_Voxellist)->as<voxellist::BitVectorVoxelList>()->
//     collideWithTypes(collisiondata->gvl->getMap(collisiondata->Robot_2_Voxellist)->as<voxellist::BitVectorVoxelList>(),bits_in_collision);
    
    collisiondata->ncoll = gvl->getMap(collisiondata->Robot_1_Voxellist)->as<voxellist::BitVectorVoxelList>()->
        collideWithTypes(gvl->getMap(collisiondata->Environment_Voxelmap)->as<voxelmap::BitVectorVoxelMap>(),bits_in_collision);
    
    avg_coll_time= avg_coll_time + (double)(clock()-time1)/CLOCKS_PER_SEC;
    
    if (counter==999)
    {
      std::cout << "Average Inserction map time in milliseconds " << avg_inserction_time << std::endl;
      std::cout << "Average Collision time in milliseconds " << avg_coll_time << std::endl;
      std::cout << "N collision " << collisiondata->ncoll << std::endl;
    }
    
    counter++;
    
  }
}