#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/JointState.h>
#include <tf/transform_broadcaster.h>
 
#include <sstream>
#include <fstream>
#include <string>
#include <stdio.h>
 

int main(int argc, char **argv)
{

  ros::init(argc, argv, "talker");
  ros::NodeHandle n;
 
  ros::Publisher joint_pub = n.advertise<sensor_msgs::JointState>("joint_states", 1);
  tf::TransformBroadcaster broadcaster;
  ros::Rate loop_rate(10);
  
  sensor_msgs::JointState joint_state;
  int path_pnt = atoi(argv[2]);
  
  float posjoint[6];
  float robjoint[6*path_pnt];
  
  std::ifstream infile;
  
  infile.open( std::string(argv[1]) + "/positioner.txt");

  if (infile.fail())
  {
    std::cout << "error"<< "\n";
  }
  else
  {
    float value;
    int i=0;
    
    while (infile >> value )
    {
      posjoint[i]=value;
      i++;
    }
    std::cout <<"\n";
  }
  infile.close();
  
  infile.open(std::string(argv[1]) + "/Robot.txt");

  if (infile.fail())
  {
    std::cout << "error"<< "\n";
  }
  else
  {
    float value;
    int i=0;
    
    while (infile >> value )
    {
      robjoint[i]=value;
      i++;
    }
  }

  int i=0;
  while (ros::ok())
  {
    if (i>path_pnt-1)
    {
      i=0;
    }
    
    joint_state.header.stamp = ros::Time::now();
    joint_state.name.resize(12);
    joint_state.position.resize(12);
    
    joint_state.name[0] ="pos_joint_1";
    joint_state.position[0] = posjoint[0];
    joint_state.name[1] ="pos_joint_2";
    joint_state.position[1] = posjoint[1];
    joint_state.name[2] ="pos_joint_3";
    joint_state.position[2] = posjoint[2];
    joint_state.name[3] ="pos_joint_4";
    joint_state.position[3] = posjoint[3];
    joint_state.name[4] ="pos_joint_5";
    joint_state.position[4] = posjoint[4];
    joint_state.name[5] ="pos_joint_6";
    joint_state.position[5] = posjoint[5];
    
    joint_state.name[6] ="rob_joint_1";
    joint_state.position[6] = robjoint[i*6];
    joint_state.name[7] ="rob_joint_2";
    joint_state.position[7] = robjoint[i*6+1];
    joint_state.name[8] ="rob_joint_3";
    joint_state.position[8] = robjoint[i*6+2];
    joint_state.name[9] ="rob_joint_4";
    joint_state.position[9] = robjoint[i*6+3];
    joint_state.name[10] ="rob_joint_5";
    joint_state.position[10] = robjoint[i*6+4];
    joint_state.name[11] ="rob_joint_6";
    joint_state.position[11] = robjoint[i*6+5];
    
    i++;
    
    joint_pub.publish(joint_state);

    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
 }
