#include <roswacocuda.h>

void WACOCuda::InitializeCollisions()
{
  float voxel_size=0.05;    //stessa unità di misura dell'URDF
  float map_size_x=5;
  float map_size_y=5;
  float map_size_z=3; 
  int voxel_x=map_size_x/voxel_size;
  int voxel_y=map_size_y/voxel_size;
  int voxel_z=map_size_z/voxel_size;

  float tool_voxel_size=0.005;
  float tool_map_size_x=5;
  float tool_map_size_y=5;
  float tool_map_size_z=3;
  int tool_voxel_x=map_size_x/tool_voxel_size;
  int tool_voxel_y=map_size_y/tool_voxel_size;
  int tool_voxel_z=map_size_z/tool_voxel_size;

  collisiondata.Robot1="Robot1";
  collisiondata.Robot2="Robot2";
  collisiondata.Environment="Environment";
  
  collisiondata.Robot1JointNames.resize(Kine_param.JOINTS_POS);
  collisiondata.Robot2JointNames.resize(Kine_param.JOINTS_TOOL);
  
  for(int posjn=0;posjn < Kine_param.JOINTS_POS;posjn++)
  {
    collisiondata.Robot1JointNames[posjn] = _PosJointsNames[posjn];
  }
  
  for(int robjn=0;robjn < Kine_param.JOINTS_POS;robjn++)
  {
    collisiondata.Robot2JointNames[robjn] = _RobJointsNames[robjn];
  }
  
  collisiondata.Robot_1_Voxellist = "Robot_1_Voxellist";
  collisiondata.Robot_2_Voxellist = "Robot_2_Voxellist";
  collisiondata.Environment_Voxelmap = "Environment_Voxelmap";

  gvl = GpuVoxels::getInstance();
  gvl->initialize(voxel_x, voxel_y, voxel_z, voxel_size); // map of x-y-z voxels each one of voxels_size dimension

  gvl->addMap(MT_BITVECTOR_VOXELLIST, collisiondata.Robot_2_Voxellist);
  gvl->addMap(MT_BITVECTOR_VOXELLIST, collisiondata.Robot_1_Voxellist);
  
  gvl->addMap(gpu_voxels::MT_BITVECTOR_VOXELMAP, collisiondata.Environment_Voxelmap);
  
  gvl->addRobot(collisiondata.Robot1, _urdf_paths[0], false);
  gvl->addRobot(collisiondata.Robot2, _urdf_paths[1], false);
  gvl->addRobot(collisiondata.Environment, _urdf_paths[2], false);
  
  gvl->insertRobotIntoMap(collisiondata.Environment, collisiondata.Environment_Voxelmap, eBVM_OCCUPIED);
  
  
  tool_gripper.Robot1 = "Gripper";
  tool_gripper.Robot2 = "Tool";
  tool_gripper.Robot1JointNames.resize(Kine_param.JOINTS_POS);
  tool_gripper.Robot2JointNames.resize(Kine_param.JOINTS_TOOL);
  
  for(int posjn=0;posjn < Kine_param.JOINTS_POS;posjn++)
  {
    tool_gripper.Robot1JointNames[posjn] = _PosJointsNames[posjn];
  }
  
  for(int robjn=0;robjn < Kine_param.JOINTS_POS;robjn++)
  {
    tool_gripper.Robot2JointNames[robjn] = _RobJointsNames[robjn];
  }

  gripper_workpiece = new voxellist::BitVectorVoxelList(gpu_voxels::Vector3ui(tool_voxel_x, tool_voxel_y, tool_voxel_z), tool_voxel_size, MT_BITVECTOR_VOXELLIST);
  tool = new voxellist::BitVectorVoxelList(gpu_voxels::Vector3ui(tool_voxel_x, tool_voxel_y, tool_voxel_z), tool_voxel_size, MT_BITVECTOR_VOXELLIST);
  
  gvl->addRobot(tool_gripper.Robot1, _urdf_paths[3], false);
  gvl->addRobot(tool_gripper.Robot2, _urdf_paths[4], false);

}

void WACOCuda::CollisionCheck_afterACO()
#define JOINTS_POS     Kine_param.JOINTS_POS
#define JOINTS_TOOL     Kine_param.JOINTS_TOOL
#define MAX_RID_CONF    Kine_param.MAX_RID_CONF
#define PATH_PNT        Kine_param.PATH_PNT
#define Z_AXIS_DISCR    Kine_param.Z_AXIS_DISCR
#define RID_CONF        Kine_param.RID_CONF
{
  int pnt_feasible=0;
  Multi_Graph_Helper Graph(hostgraph,optim_param.WOA.whales,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL);
  Multi_Positioner_Helper Positioner(hostjointpos,optim_param.WOA.whales,JOINTS_POS);
  int path_feasibility=0;
  
  bool coll_positioner_env=false;
  bool coll_gripper_tool=false;
  bool coll_toolrob_env=false;
  bool coll_posit_toolrob=false;
  bool feasible=true;
  
  no_collisions=false;
  
  int i_whale=check_whale_coll;
  collisiondata.ncoll=0;

  if (i_whale>=0)
  {
    if ( hostcheckreach[i_whale]==true && already_checked_coll[i_whale]==false )
    {
      coll_positioner_env =  Collision_Positioner_Environment(Positioner, i_whale, Graph);
    }
  }
  else
  { 
    return;
  }
  
  if (coll_positioner_env==false)
  {
    hostcheckreach[i_whale]=false;
    already_checked_coll[i_whale]=true;
    host_maxFObjWhales[i_whale]=-1.0;
    
    return;
  }
  
  if (coll_positioner_env)
  {
    coll_gripper_tool = Collision_Gripper_Tool(Positioner,Graph,i_whale,&feasible, toolrob_gripp_check);
  }
  
  if (feasible==false)
  {
    hostcheckreach[i_whale]=false;
    already_checked_coll[i_whale]=true;
    host_maxFObjWhales[i_whale]=-1.0;
    
    return;
  }
  
  if (coll_gripper_tool)
  {
    coll_toolrob_env = Collision_Toolrobot_Environment(Graph,i_whale,&feasible, toolrob_env_check);
  }
  
  if (feasible==false)
  {
    hostcheckreach[i_whale]=false;
    already_checked_coll[i_whale]=true;
    host_maxFObjWhales[i_whale]=-1.0;
    
    return;
  }
  
  if (coll_toolrob_env)
  {
    coll_posit_toolrob = Collision_Positioner_Toolrobot(Positioner, Graph, i_whale,&feasible, toolrob_posit_check);
  }
  

  if (coll_posit_toolrob)
  {
    already_checked_coll[i_whale]=true;
    no_collisions=true;
  }
  
  if (feasible==false)
  {
    hostcheckreach[i_whale]=false;
    already_checked_coll[i_whale]=true;
    host_maxFObjWhales[i_whale]=-1.0;
    
    return;
  }
  
#undef JOINTS_POS
#undef JOINTS_TOOL
#undef MAX_RID_CONF
#undef PATH_PNT
#undef Z_AXIS_DISCR
#undef RID_CONF
}

bool WACOCuda::Collision_Positioner_Environment(Multi_Positioner_Helper Positioner, int i_whale, Multi_Graph_Helper Graph)
{
  BitVectorVoxel bits_in_collision;
  bool collision_check=true;
  collisiondata.ncoll=0;
  
  Update_positioner_gpu_voxels(Positioner,i_whale);
  
  collisiondata.ncoll = gvl->getMap(collisiondata.Robot_1_Voxellist)->as<voxellist::BitVectorVoxelList>()->
        collideWithTypes(gvl->getMap(collisiondata.Environment_Voxelmap)->as<voxelmap::BitVectorVoxelMap>(),bits_in_collision);
  
  if (collisiondata.ncoll>0)
  {
    collision_check=false;
    for (int pnt=0; pnt<Kine_param.PATH_PNT; pnt++)
    {
      for (int conf=0; conf<Kine_param.MAX_OPEN_CONE_ANGLE; conf++)
      {
        Graph.setUnfeas(i_whale,pnt,conf);
      }
    }
    
    hostcheckreach[i_whale]=false;
    already_checked_coll[i_whale]=true;
    host_maxFObjWhales[i_whale]=-1.0;
  }
  return collision_check;
}

bool WACOCuda::Collision_Gripper_Tool(Multi_Positioner_Helper Positioner, Multi_Graph_Helper Graph,int i_whale, bool* feasible,
                                      bool* check_coll_all_points)
{
  BitVectorVoxel bits_in_collision;
  bool collision_check=true;
  int counter=0;
  
  check_coll_all_points=check_coll_all_points + i_whale*Kine_param.PATH_PNT;
  
  Update_gripper_gpu_voxels(Positioner,i_whale);
        
  for (int pnt=0; pnt<Kine_param.PATH_PNT; pnt++)
  {
    if (!check_coll_all_points[pnt])
    {
      int config = host_ants_path[i_whale*Kine_param.PATH_PNT + pnt];
      
      Update_tool_gpu_voxels(Graph,i_whale,pnt,config);
      
      tool_gripper.ncoll = gripper_workpiece->collideWithTypes(tool, bits_in_collision);
      
      if (tool_gripper.ncoll>0)
      {
        collision_check=false;
        Graph.setUnfeas(i_whale,pnt, config);
        
        for( int pose=0; pose<Kine_param.MAX_RID_CONF_REAL; pose++)
        {
          if (Graph.getCh(i_whale,pnt, pose) )
          {
            Update_tool_gpu_voxels(Graph,i_whale,pnt,pose);
            
            tool_gripper.ncoll = gripper_workpiece->collideWithTypes(tool, bits_in_collision);
          
            if (tool_gripper.ncoll>0)
            {
              Graph.setUnfeas(i_whale,pnt, pose); 
            }
            else
            {
              counter++;
            }
          }
        }
        if(counter==0)
        {
          feasible[0]=false;
          return collision_check;
        }
        else
        {
          counter=0;
          check_coll_all_points[pnt]=true;
        }
        break;
      }
    }
  }
  feasible[0]=true;
  return collision_check;
}

bool WACOCuda::Collision_Toolrobot_Environment(Multi_Graph_Helper Graph,int i_whale, bool* feasible, bool* check_coll_all_points)
{
  BitVectorVoxel bits_in_collision;
  bool collision_check=true;
  int counter=0;
  collisiondata.ncoll=0;
  
  check_coll_all_points=check_coll_all_points + i_whale*Kine_param.PATH_PNT;
  
  for (int pnt=0; pnt<Kine_param.PATH_PNT; pnt++)
  {
    collisiondata.ncoll=0;
    
    if (!check_coll_all_points[pnt])
    {
      Update_robot_gpu_voxels(Graph,i_whale,pnt,host_ants_path[i_whale*Kine_param.PATH_PNT + pnt]);
        
      collisiondata.ncoll = gvl->getMap(collisiondata.Robot_2_Voxellist)->as<voxellist::BitVectorVoxelList>()->
            collideWithTypes(gvl->getMap(collisiondata.Environment_Voxelmap)->as<voxelmap::BitVectorVoxelMap>(),bits_in_collision);
      
      if (collisiondata.ncoll>0)
      {
        collision_check=false;
        Graph.setUnfeas(i_whale,pnt,host_ants_path[i_whale*Kine_param.PATH_PNT + pnt]);
        
        for (int conf=0; conf< Kine_param.MAX_RID_CONF_REAL; conf++)
        {
          if ( Graph.getCh(i_whale,pnt,conf) )
          {          
            collisiondata.ncoll=0;

            Update_robot_gpu_voxels(Graph,i_whale,pnt,conf);
            collisiondata.ncoll = gvl->getMap(collisiondata.Robot_2_Voxellist)->as<voxellist::BitVectorVoxelList>()->
            collideWithTypes(gvl->getMap(collisiondata.Environment_Voxelmap)->as<voxelmap::BitVectorVoxelMap>(),bits_in_collision);
            
            if (collisiondata.ncoll>0)
            {
              Graph.setUnfeas(i_whale,pnt,conf);
            }
            else
            {
              counter++;
            }
          }
          collisiondata.ncoll=0;
        }
        if (counter==0)
        {
          feasible[0]=false;
          return collision_check;
        }
        else
        {
          counter=0;
          check_coll_all_points[pnt]=true;
        }
        break;
      }
    }
  }
  feasible[0]=true;
  return collision_check;
}

bool WACOCuda::Collision_Positioner_Toolrobot(Multi_Positioner_Helper Positioner, Multi_Graph_Helper Graph, int i_whale, bool* feasible,
                                              bool* check_coll_all_points)
{
  BitVectorVoxel bits_in_collision;
  bool collision_check=true;
  int counter=0;
  collisiondata.ncoll=0;
  
  check_coll_all_points=check_coll_all_points + i_whale*Kine_param.PATH_PNT;
  
  for (int pnt=0; pnt<Kine_param.PATH_PNT; pnt++)
  {
    collisiondata.ncoll=0;
    
    if (!check_coll_all_points[pnt])
    {
      Update_robot_gpu_voxels(Graph,i_whale,pnt,host_ants_path[i_whale*Kine_param.PATH_PNT + pnt]);
      
      Update_positioner_gpu_voxels(Positioner,i_whale);
      
      collisiondata.ncoll = gvl->getMap(collisiondata.Robot_1_Voxellist)->as<voxellist::BitVectorVoxelList>()->
                          collideWithTypes(gvl->getMap(collisiondata.Robot_2_Voxellist)->as<voxellist::BitVectorVoxelList>(),
                          bits_in_collision);
                          
      if (collisiondata.ncoll>0)
      {
        collision_check=false;
        Graph.setUnfeas(i_whale,pnt,host_ants_path[i_whale*Kine_param.PATH_PNT + pnt]);
        
        for (int conf=0; conf<Kine_param.MAX_RID_CONF_REAL; conf++)
        {
          if ( Graph.getCh(i_whale,pnt,conf) )
          {
            Update_robot_gpu_voxels(Graph,i_whale,pnt,conf);
            
            collisiondata.ncoll = gvl->getMap(collisiondata.Robot_1_Voxellist)->as<voxellist::BitVectorVoxelList>()->
                          collideWithTypes(gvl->getMap(collisiondata.Robot_2_Voxellist)->as<voxellist::BitVectorVoxelList>(),
                          bits_in_collision);
            
            if (collisiondata.ncoll>0)
            {
              Graph.setUnfeas(i_whale,pnt,conf);
            }
            else
            {
              counter++;
            }
          }
        }
        if (counter==0)
        {
          feasible[0]=false;
          return collision_check;
        }
        else
        {
          counter=0;
          check_coll_all_points[pnt]=true;
        }
        break;
      }
    }
  }
  feasible[0]=true;
  return collision_check;
}

//////////////////////MACRO///////////////////////////////////////////////////////////

void WACOCuda::Update_gripper_gpu_voxels(Multi_Positioner_Helper Positioner, int whale)
{

  RobotInterfaceSharedPtr gripper_pointer = gvl->getRobot(tool_gripper.Robot1);
  gripper_workpiece->clearMap();
  
  for (int jnt=0; jnt<Kine_param.JOINTS_POS; jnt++)
  {
    tool_gripper.myRobotJointValues[tool_gripper.Robot1JointNames[jnt]] = Positioner(whale, jnt);
  }
  
  gvl->setRobotConfiguration(tool_gripper.Robot1, tool_gripper.myRobotJointValues);
  gripper_workpiece->insertMetaPointCloud(*(gripper_pointer->getTransformedClouds()),eBVM_OCCUPIED);

}

void WACOCuda::Update_tool_gpu_voxels(Multi_Graph_Helper Graph, int whale ,int pnt, int config)
{

  tool->clearMap();
  RobotInterfaceSharedPtr tool_pointer = gvl->getRobot(tool_gripper.Robot2);
  
  for (int jnt=0; jnt<Kine_param.JOINTS_TOOL; jnt++)
  {
    tool_gripper.myRobotJointValues2[tool_gripper.Robot2JointNames[jnt]] = Graph(whale,pnt,config,jnt);
  }
  gvl->setRobotConfiguration(tool_gripper.Robot2, tool_gripper.myRobotJointValues2);
  tool->insertMetaPointCloud(*(tool_pointer->getTransformedClouds()),eBVM_OCCUPIED);
}

 void WACOCuda::Update_positioner_gpu_voxels(Multi_Positioner_Helper Positioner, int whale)
{
  gvl->GpuVoxels::clearMap(collisiondata.Robot_1_Voxellist);
  for (int jnt=0; jnt<Kine_param.JOINTS_POS; jnt++)
  {
    collisiondata.myRobotJointValues[collisiondata.Robot1JointNames[jnt]] = Positioner(whale,jnt);
  }
  
  gvl->setRobotConfiguration(collisiondata.Robot1, collisiondata.myRobotJointValues);
  gvl->insertRobotIntoMap(collisiondata.Robot1, collisiondata.Robot_1_Voxellist, eBVM_OCCUPIED);
}

void WACOCuda::Update_robot_gpu_voxels(Multi_Graph_Helper Graph, int whale, int pnt,int rid_config)
{
  gvl->GpuVoxels::clearMap(collisiondata.Robot_2_Voxellist);
  for (int jnt=0; jnt<Kine_param.JOINTS_TOOL; jnt++)
  {
    collisiondata.myRobotJointValues2[collisiondata.Robot2JointNames[jnt]] = Graph(whale,pnt,rid_config,jnt);
  }

  gvl->setRobotConfiguration(collisiondata.Robot2, collisiondata.myRobotJointValues2);
  gvl->insertRobotIntoMap(collisiondata.Robot2, collisiondata.Robot_2_Voxellist, eBVM_OCCUPIED);
}


