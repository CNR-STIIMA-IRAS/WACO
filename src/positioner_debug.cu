#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/JointState.h>
#include <tf/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <roswacocuda.h>
#include <cuda.h>

#define IK_VERSION 61
#define IKFAST_NO_MAIN true
#define IKFAST_HAS_LIBRARY

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
 
#include <sstream>
#include <fstream>
#include <string>
#include <stdio.h>
 

int main(int argc, char **argv)
{

  int whales = atoi(argv[2]);
  int cycles = atoi(argv[3]);
  
  ros::init(argc, argv, "talker");
  ros::NodeHandle n;
 
//   tf2_ros::TransformBroadcaster broadcaster;
//   geometry_msgs::TransformStamped transformStamped;
  
  tf2_ros::StaticTransformBroadcaster broadcaster;
  geometry_msgs::TransformStamped transformStamped;
  
  ros::Rate loop_rate(1);
  
  Eigen::Matrix4f Pos_TCP;
  positioner_robot::IkReal eerot[9],eetrans[3];
  positioner_robot::IkReal* positioner_joints = new positioner_robot::IkReal[whales * cycles * 6];

  
  std::ifstream infile;
  std::string path("/home/kolmogorov/Documents/ROS/waco/logs/" + std::string(argv[1]) + "/positioner.txt");
  infile.open(path);
  infile.seekg(0, std::ios::beg);
//   infile.open("/home/kolmogorov/Documents/ROS/waco/logs/" + std::string(argv[1]) + "/positioner.txt", std::ios::in);
  
  if (infile.fail())
  {
    std::cout << "error"<< "\n";
  }
  else
  {
    float value;
    int i=0;
    
    while (infile >> value)
    {
      positioner_joints[i]=value;
      i++;
    }
  }
  infile.close();

  Eigen::Affine3f aff;
  
  for(int cy=0;cy<whales*cycles;cy++)
  {
    positioner_robot::ComputeFk((positioner_joints + (cy *6)),eetrans,eerot);
    
    Pos_TCP(0,0)=eerot[0];Pos_TCP(0,1)=eerot[1];Pos_TCP(0,2)=eerot[2];Pos_TCP(0,3)=eetrans[0];
    Pos_TCP(1,0)=eerot[3];Pos_TCP(1,1)=eerot[4];Pos_TCP(1,2)=eerot[5];Pos_TCP(1,3)=eetrans[1];
    Pos_TCP(2,0)=eerot[6];Pos_TCP(2,1)=eerot[7];Pos_TCP(2,2)=eerot[8];Pos_TCP(2,3)=eetrans[2];
    Pos_TCP(3,0)=0       ;Pos_TCP(3,1)=0       ;Pos_TCP(3,2)=0       ;Pos_TCP(3,3)=1;
    aff.matrix() = Pos_TCP;
    Eigen::Affine3d tmp=aff.cast<double>();
//     std::cout << (positioner_joints + (cy *6))[0] << " "
//               << (positioner_joints + (cy *6))[1] << " "
//               << (positioner_joints + (cy *6))[2] << " "
//               << (positioner_joints + (cy *6))[3] << " "
//               << (positioner_joints + (cy *6))[4] << " "
//               << (positioner_joints + (cy *6))[5] << " " << std::endl<<std::endl;
//     std::cout << tmp.matrix()<<std::endl<<std::endl;
    transformStamped = tf2::eigenToTransform(tmp);
    transformStamped.header.frame_id="world";
    transformStamped.child_frame_id="pippo" + std::to_string(cy);
    broadcaster.sendTransform(transformStamped);
  }
  
  while (ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}