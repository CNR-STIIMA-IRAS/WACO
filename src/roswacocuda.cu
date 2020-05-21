#include <roswacocuda.h>
#include <cuda_functions.h>
#include <save_data.h>
#include <stdio.h>
#include <urdf/model.h>
#include <urdf_model/types.h>
#include "iostream"
#include "fstream"
#include "string.h"
#include "halton.hpp"
#include "halton.cpp"
// #include <boost/filesystem.h>


///////////WACO OPT CLASS METHODS
void WACOCuda::AllocMem(int n_whales,int PATH_PNT,int MAX_RID_CONF_REAL)
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  
  size_t  free;
  size_t  total;
  
  check_mem_ =cudaMemGetInfo(&free, &total);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  std::cout << "Free memory: " << free/(1024*1024)<<"MB\n"; 
    
//   if(Multi_Heur_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF_REAL).getsizeMulti()< free)TODO
  if(true)
  {
    std::cout << "Heuristic off-line\n";
    std::cout << "Heuristic size : " << Multi_Heur_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF_REAL).getsizeMulti()/(1024*1024) << "MB\n";
    
    heuristic_online=false;
    
    heuristichost_ = malloc(Multi_Heur_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF_REAL).getsizeMulti());
    memset(heuristichost_,0x0,Multi_Heur_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF_REAL).getsizeMulti());
    check_mem_ = cudaMalloc(static_cast<void**>(&heuristicdev_),Multi_Heur_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF_REAL).getsizeMulti());
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    check_mem_ = cudaMemcpy(heuristicdev_,heuristichost_,Multi_Heur_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF_REAL).getsizeMulti(),cudaMemcpyHostToDevice);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  }
  else
  {
    std::cout << "Heuristic online\n";
    std::cout << "Total Memory Heu Online: " << Multi_Heur_Helper(0x0,n_whales,1,MAX_RID_CONF_REAL).getsizeMulti()/(1024*1024) << " MB\n";
    
    heuristic_online=true;
    
    heuristichost_ = malloc(Multi_Heur_Helper(0x0,n_whales,1,MAX_RID_CONF_REAL).getsizeMulti());
    memset(heuristichost_,0x0,Multi_Heur_Helper(0x0,n_whales,1,MAX_RID_CONF_REAL).getsizeMulti());
    
    check_mem_ = cudaMalloc(static_cast<void**>(&heuristicdev_),Multi_Heur_Helper(0x0,n_whales,1,MAX_RID_CONF_REAL).getsizeMulti());
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    
  }
}

WACOCuda::WACOCuda(int n_whales,int cyc_w,const boundaries& robot1, const boundaries& robot2, float A_lim, trajectory* pathex,
                   Eigen::Matrix4f RobotsTransex, Eigen::Matrix4f GrabTransex, int n_ants, int ant_cycles, Kine_const* KINE_param,
                   Eigen::Matrix4f RobotTCP_tool,float beta,std::vector<std::string> PosJointsNames,std::vector<std::string> RobJointsNames,std::vector<std::string> urdf_paths)
                   : robot_pos_limits( robot1 ), robot_tool_limits( robot2 ),
                   _PosJointsNames(PosJointsNames), _RobJointsNames(RobJointsNames),_urdf_paths(urdf_paths)
{
  check_mem_ = cudaDeviceSetLimit(cudaLimitMallocHeapSize,(1024*1024*1024));//forse troppo
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaDeviceSetLimit(cudaLimitStackSize,65536); //??????
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  int PATH_PNT = pathex->n_points;
  int MAX_RID_CONF = KINE_param->MAX_RID_CONF;
  int JOINTS_POS  = KINE_param->JOINTS_POS;
  int JOINTS_TOOL = KINE_param->JOINTS_TOOL;
  int Z_AXIS_DISCR = KINE_param->Z_AXIS_DISCR;
  int RID_CONF = KINE_param->RID_CONF;
  
  check_whale_coll=-1;
  
  optim_param.WOA.whales=n_whales;
  optim_param.WOA.cycles=cyc_w;
  optim_param.WOA.A_lim=A_lim;
  
  optim_param.ACO.ants=n_ants;
  optim_param.ACO.cycles=ant_cycles;
  optim_param.ACO.shr_bytes_ACO= sizeof(float)*4 + 8*sizeof(int) + sizeof(float)*3*n_ants;
  
  optim_param.ACO.beta = beta;
    
#if defined(DEBUG_HEUR) || defined(DEBUG_HEUR_POS)
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
#endif
  
  Kine_param=*KINE_param;
  inv_tool_trans = RobotTCP_tool.inverse();

  float *hal1 = new float[(int)KINE_param->Z_AXIS_DISCR];
  float *hal2 = new float[(int)KINE_param->N_OPEN_CONE];
  float *hal3 = new float[(int)KINE_param->N_ROT_CONE];

  int b1[1] = {7};
  int b2[1] = {3};
  int b3[1] = {2};

  for (int pd=0;pd<KINE_param->Z_AXIS_DISCR;pd++)
  {
      hal1[pd] = (float)*(halton_base(pd,1,b1));
  }

  for (int pd=0;pd<KINE_param->N_OPEN_CONE;pd++)
  {
      hal2[pd] = (float)*(halton_base(pd,1,b2));
  }

  for (int pd=0;pd<KINE_param->N_ROT_CONE;pd++)
  {
      hal3[pd] = (float)*(halton_base(pd,1,b3));
  }

  std::sort(hal1,hal1+(int)KINE_param->Z_AXIS_DISCR,std::less<float>());
  std::sort(hal2,hal2+(int)KINE_param->N_OPEN_CONE,std::less<float>());
  std::sort(hal3,hal3+(int)KINE_param->N_ROT_CONE,std::less<float>());

  std::cout << "HAL1"  << std::endl;
  for (int pd=0;pd<KINE_param->Z_AXIS_DISCR;pd++)
  {
      std::cout << hal1[pd] << " ";
  }

  std::cout << std::endl << "HAL2"  << std::endl;
  for (int pd=0;pd<KINE_param->N_OPEN_CONE;pd++)
  {
      std::cout << hal2[pd] << " ";
  }

  std::cout << std::endl << "HAL3"  << std::endl;
  for (int pd=0;pd<KINE_param->N_ROT_CONE;pd++)
  {
      std::cout << hal3[pd] << " ";
  }

  std::cout << std::endl;
  
  /////MEMORY INITIALIZATION 
  ///
    check_mem_ = cudaMalloc(static_cast<float**>(&hal1_dev),((int)KINE_param->Z_AXIS_DISCR)*sizeof(float));
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    check_mem_ = cudaMemcpy(hal1_dev,hal1,((int)KINE_param->Z_AXIS_DISCR)*sizeof(float),cudaMemcpyHostToDevice);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);

    check_mem_ = cudaMalloc(static_cast<float**>(&hal2_dev),((int)KINE_param->N_OPEN_CONE)*sizeof(float));
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    check_mem_ = cudaMemcpy(hal2_dev,hal2,((int)KINE_param->N_OPEN_CONE)*sizeof(float),cudaMemcpyHostToDevice);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);

    check_mem_ = cudaMalloc(static_cast<float**>(&hal3_dev),((int)KINE_param->N_ROT_CONE)*sizeof(float));
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    check_mem_ = cudaMemcpy(hal3_dev,hal3,((int)KINE_param->N_ROT_CONE)*sizeof(float),cudaMemcpyHostToDevice);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);


  fobj_history=(float*) malloc(n_whales*cyc_w*sizeof(float));
  std::fill_n(fobj_history,n_whales*cyc_w,0.0);
  
  toolrob_env_check=(bool*) malloc(n_whales*PATH_PNT*sizeof(bool));
  std::fill_n(toolrob_env_check,n_whales*PATH_PNT,false);
  toolrob_gripp_check=(bool*) malloc(n_whales*PATH_PNT*sizeof(bool));
  std::fill_n(toolrob_gripp_check,n_whales*PATH_PNT,false);
  toolrob_posit_check=(bool*) malloc(n_whales*PATH_PNT*sizeof(bool));
  std::fill_n(toolrob_posit_check,n_whales*PATH_PNT,false);
  
  hostgraph = malloc(Multi_Graph_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph());
  hostgraphempty = malloc(Multi_Graph_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph());
  memset(hostgraph,0x0,Multi_Graph_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph());
  memset(hostgraphempty,0x0,Multi_Graph_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph());
  
  check_mem_ = cudaMalloc(static_cast<void**>(&devgraph),Multi_Graph_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph());
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devgraph,hostgraph,Multi_Graph_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<double**>(&solIK),optim_param.ACO.ants*sizeof(double)*JOINTS_TOOL*n_whales);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);   
//   check_mem_ = cudaMemset(solIK,0,optim_param.ACO.ants*sizeof(double)*JOINTS_TOOL*n_whales);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  max_conf_graph_host = (int*)malloc(n_whales * sizeof(int));
  memset(max_conf_graph_host,0x0,n_whales * sizeof(int));
  check_mem_ = cudaMalloc(static_cast<int**>(&max_conf_graph_dev),n_whales * sizeof(int));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(max_conf_graph_dev,max_conf_graph_host,n_whales * sizeof(int),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  hostsoltraj = malloc(Multi_Graph_Helper(0x0,1,PATH_PNT,1,JOINTS_TOOL).getSizeMultiGraph());
  memset(hostsoltraj,0x0,Multi_Graph_Helper(0x0,1,PATH_PNT,1,JOINTS_TOOL).getSizeMultiGraph());
  check_mem_ = cudaMalloc(static_cast<void**>(&devsoltraj),Multi_Graph_Helper(0x0,1,PATH_PNT,1,JOINTS_TOOL).getSizeMultiGraph());
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devsoltraj,hostsoltraj,Multi_Graph_Helper(0x0,1,PATH_PNT,1,JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  hostjointbest = malloc(Multi_Positioner_Helper(0x0,1,JOINTS_POS).getsizeMulti());
  memset(hostjointbest,0x0,Multi_Positioner_Helper(0x0,1,JOINTS_POS).getsizeMulti());
  check_mem_ = cudaMalloc(static_cast<void**>(&devicejointbest),Multi_Positioner_Helper(0x0,1,JOINTS_POS).getsizeMulti());
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devicejointbest,hostjointbest,Multi_Positioner_Helper(0x0,1,JOINTS_POS).getsizeMulti(),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  host_maxFObjWhales = (float*)malloc(n_whales*sizeof(float));
  std::fill_n(host_maxFObjWhales,n_whales,-1.0);
  check_mem_ = cudaMalloc(static_cast<float**>(&dev_maxFObjWhales),n_whales*sizeof(float));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_maxFObjWhales,host_maxFObjWhales,n_whales*sizeof(float),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  hostBestscore = (float*)malloc(sizeof(float));
  hostBestscore[0] = -1;
  check_mem_ = cudaMalloc(static_cast<float**>(&deviceBestscore),sizeof(float));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(deviceBestscore,hostBestscore,sizeof(float),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  hostcheckreach = (bool*)malloc(n_whales*sizeof(bool));
  std::fill_n(hostcheckreach,n_whales,true);
  check_mem_ = cudaMalloc(static_cast<bool**>(&devcheckreach),sizeof(bool)*n_whales);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devcheckreach,hostcheckreach,sizeof(bool)*n_whales,cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  hostjointpos = malloc(Multi_Positioner_Helper(0x0,n_whales,JOINTS_POS).getsizeMulti());
  memset(hostjointpos,0x0,Multi_Positioner_Helper(0x0,n_whales,JOINTS_POS).getsizeMulti());
  check_mem_ = cudaMalloc(static_cast<void**>(&devjointpos),Multi_Positioner_Helper(0x0,n_whales,PATH_PNT).getsizeMulti());
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devjointpos,hostjointpos,Multi_Positioner_Helper(0x0,n_whales,PATH_PNT).getsizeMulti(),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  host_ants_path = (int*)malloc(PATH_PNT*n_whales*sizeof(int));
  std::fill_n(host_ants_path,PATH_PNT*n_whales,-1);
  check_mem_ = cudaMalloc(static_cast<int**>(&dev_ants_path),PATH_PNT*n_whales*sizeof(int));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_ants_path,host_ants_path,PATH_PNT*n_whales*sizeof(int),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  already_checked_coll = (bool*)malloc(n_whales*sizeof(bool));//TODO
  std::fill_n(already_checked_coll,n_whales,false);
  check_mem_ = cudaMalloc(static_cast<bool**>(&dev_already_checked_coll),n_whales*sizeof(bool));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_already_checked_coll,already_checked_coll,n_whales*sizeof(bool),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  hostpathsol = (Eigen::Matrix4f*)malloc(PATH_PNT*sizeof(Eigen::Matrix4f));
  for(int mat=0;mat<PATH_PNT;mat++) hostpathsol[mat] = Eigen::Matrix4f::Zero();
  check_mem_ = cudaMalloc(static_cast<Eigen::Matrix4f**>(&devpathsol),sizeof(Eigen::Matrix4f)*PATH_PNT);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devpathsol,hostpathsol,sizeof(Eigen::Matrix4f)*PATH_PNT,cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMallocHost((void**)&host_pos_tcp,n_whales*sizeof(Eigen::Matrix4f));
  for(int mat=0;mat<n_whales;mat++) host_pos_tcp[mat] = Eigen::Matrix4f::Zero();
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<int**>(&soldev),n_whales*PATH_PNT*n_ants*sizeof(int));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemset(soldev,0,n_whales*PATH_PNT*n_ants*sizeof(int));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<joint_boundary**>(&dev_robot_pos_lim_joint),robot_pos_limits.n_joints * sizeof(joint_boundary) );
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_robot_pos_lim_joint,robot_pos_limits.pos_joint,robot_pos_limits.n_joints*sizeof(joint_boundary),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<joint_boundary**>(&dev_robot_tool_lim_joint),robot_tool_limits.n_joints * sizeof(joint_boundary) );
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_robot_tool_lim_joint,robot_tool_limits.pos_joint,robot_tool_limits.n_joints*sizeof(joint_boundary),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);

  check_mem_ = cudaMalloc(static_cast<joint_boundary**>(&dev_robot_pos_speed_joint),robot_pos_limits.n_joints * sizeof(joint_boundary) );
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_robot_pos_speed_joint,robot_pos_limits.speed_joint,robot_pos_limits.n_joints*sizeof(joint_boundary),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<joint_boundary**>(&dev_robot_tool_speed_joint),robot_tool_limits.n_joints * sizeof(joint_boundary) );
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_robot_tool_speed_joint,robot_tool_limits.speed_joint,robot_tool_limits.n_joints*sizeof(joint_boundary),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);

  check_mem_ = cudaMalloc(static_cast<int**>(&dev_multiturn_axis),JOINTS_TOOL*sizeof(int));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_multiturn_axis,robot_tool_limits.multiturn_bool,JOINTS_TOOL*sizeof(int),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<Eigen::Matrix4f**>(&dev_pathrel),PATH_PNT * sizeof(Eigen::Matrix4f));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_pathrel,pathex->points,PATH_PNT * sizeof(Eigen::Matrix4f),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<float**>(&dev_timevector),(PATH_PNT-1)*sizeof(float));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_timevector,pathex->timevector,(PATH_PNT-1)*sizeof(float), cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<Eigen::Matrix4f**>(&dev_pos_tcp),n_whales*sizeof(Eigen::Matrix4f));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<Eigen::Matrix4f**>(&dev_Robot_Gripper_tcp),n_whales*sizeof(Eigen::Matrix4f));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<float**>(&dev_real_joint_angle),n_whales*n_ants*robot_tool_limits.n_joints*sizeof(float));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemset(dev_real_joint_angle,0,n_whales*n_ants*robot_tool_limits.n_joints*sizeof(float));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  check_mem_ = cudaMalloc(static_cast<Eigen::Matrix4f**>(&dev_inv_tool_trans),sizeof(Eigen::Matrix4f));
  check_mem_ = cudaMemcpy(dev_inv_tool_trans,&inv_tool_trans,sizeof(Eigen::Matrix4f),cudaMemcpyHostToDevice);
  
 
  //ANTS log
#if defined(DEBUG_HEUR) || defined(DEBUG_HEUR_POS)
  logphhost   =(float*) malloc(3*optim_param.ACO.cycles*sizeof(float)*n_whales +n_whales*PATH_PNT*MAX_RID_CONF*optim_param.ACO.cycles*sizeof(float));
  std::fill_n(logphhost,n_whales*PATH_PNT*MAX_RID_CONF*optim_param.ACO.cycles + 3*optim_param.ACO.cycles*n_whales,-1.0);
  check_mem_ = cudaMalloc(static_cast<float**>(&logphdev),3*optim_param.ACO.cycles*sizeof(float)*n_whales + n_whales*PATH_PNT*MAX_RID_CONF*optim_param.ACO.cycles*sizeof(float));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(logphdev,logphhost,3*optim_param.ACO.cycles*sizeof(float)*n_whales + n_whales*PATH_PNT*MAX_RID_CONF*optim_param.ACO.cycles*sizeof(float),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  logantshost =(int*) malloc(n_whales*n_ants*PATH_PNT*optim_param.ACO.cycles*sizeof(int));
  std::fill_n(logantshost,n_whales*n_ants*PATH_PNT*optim_param.ACO.cycles,-1);

  check_mem_ = cudaMalloc(static_cast<int**>(&logantsdev),n_whales*n_ants*PATH_PNT*optim_param.ACO.cycles*sizeof(int));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(logantsdev,logantshost,n_whales*n_ants*PATH_PNT*optim_param.ACO.cycles*sizeof(int),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
#endif
  ///
  
  check_mem_ = cudaDeviceSynchronize();
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  RobotsTrans   = RobotsTransex;
  gripper_trans = GrabTransex;
  
  this->InitializeCollisions();
  
  
  printf("Whale number: %d \n",n_whales);
  printf("Whale cycles: %d \n",cyc_w);
  printf("Ant number  : %d \n",n_ants);
  printf("Ant cycles  : %d \n",ant_cycles);
  printf("Shared bytes ants : %zu \n", optim_param.ACO.shr_bytes_ACO);
  printf("Trajectory %lu B \n",(unsigned long)sizeof(trajectory));
  printf("Max graph size for single whale  %f MB \n",((float)Multi_Graph_Helper(0x0,1,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph())/(1024.0*1024.0));
  printf("Number of configurations : %d\n",MAX_RID_CONF);
  printf("Max total memory data : %f \n",(((float)Multi_Heur_Helper(0x0,n_whales,PATH_PNT,MAX_RID_CONF).getsizeMulti())/(1024.0*1024.0)));
}

void WACOCuda::Copytohost()
#define N_WHALES      optim_param.WOA.whales
#define PATH_PNT      Kine_param.PATH_PNT
#define JOINTS_TOOL   Kine_param.JOINTS_TOOL
#define JOINTS_POS    Kine_param.JOINTS_POS
#define MAX_RID_CONF  Kine_param.MAX_RID_CONF
#define Z_AXIS_DISCR  Kine_param.Z_AXIS_DISCR
#define RID_CONF      Kine_param.RID_CONF
{
  check_mem_ = cudaMemcpy(hostBestscore,deviceBestscore,sizeof(float),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(hostjointbest,devicejointbest,Multi_Positioner_Helper(0x0,1,JOINTS_POS).getsizeMulti(),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(hostsoltraj,devsoltraj,Multi_Graph_Helper(0x0,1,PATH_PNT,1,JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaDeviceSynchronize();
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  Multi_Positioner_Helper HostJointBest(hostjointbest,1,JOINTS_POS);
  
  fprintf(stderr,"\nbest score : %f \n\n\n",*hostBestscore);
  printf("best positioner position : \n");

  printf("JNT1 : %f  : %f \n",HostJointBest(0,0),(180.0/PI_F)*HostJointBest(0,0));
  printf("JNT2 : %f  : %f \n",HostJointBest(0,1),(180.0/PI_F)*HostJointBest(0,1));
  printf("JNT3 : %f  : %f \n",HostJointBest(0,2),(180.0/PI_F)*HostJointBest(0,2));
  printf("JNT4 : %f  : %f \n",HostJointBest(0,3),(180.0/PI_F)*HostJointBest(0,3));
  printf("JNT5 : %f  : %f \n",HostJointBest(0,4),(180.0/PI_F)*HostJointBest(0,4));
  printf("JNT6 : %f  : %f \n\n\n",HostJointBest(0,5),(180.0/PI_F)*HostJointBest(0,5));
  
  Multi_Graph_Helper HostSolTraj(hostsoltraj,1,PATH_PNT,1,JOINTS_TOOL);
  
  printf("TRAJECTORY\n");
  for(int jnt=0;jnt<JOINTS_TOOL;jnt++)
  {
    for(int pnt=0;pnt<PATH_PNT;pnt++)
    {
      if (HostSolTraj(0,pnt,0,jnt)>0)
      {
        printf(" ");
      }
      printf("%f  ",HostSolTraj(0,pnt,0,jnt));
    }
    printf("\n");
  }
  
#undef JOINTS_TOOL
#undef JOINTS_POS
#undef MAX_RID_CONF
#undef PATH_PNT
#undef Z_AXIS_DISCR
#undef N_WHALES
#undef RID_CONF
}

void WACOCuda::LogAnts(std::string path,int wacyc)
#define N_WHALES      optim_param.WOA.whales
#define PATH_PNT      Kine_param.PATH_PNT
#define JOINTS_TOOL   Kine_param.JOINTS_TOOL
#define JOINTS_POS    Kine_param.JOINTS_POS
#define MAX_RID_CONF  Kine_param.MAX_RID_CONF
#define Z_AXIS_DISCR  Kine_param.Z_AXIS_DISCR
#define RID_CONF      Kine_param.RID_CONF
{
  cudaMemcpy(logantshost,logantsdev,optim_param.WOA.whales*optim_param.ACO.ants*PATH_PNT*optim_param.ACO.cycles*sizeof(int),cudaMemcpyDeviceToHost);
  
  std::ofstream log_file_ants;
  log_file_ants.open(path + "logants.txt", std::ios_base::app);

  log_file_ants << "Whale cycle : " << wacyc << std::endl << std::endl;
  for(int wa=0;wa<optim_param.WOA.whales;wa++)
  {
    log_file_ants << "Whale number : " << wa << std::endl << std::endl;
    for(int antcy=0;antcy<optim_param.ACO.cycles;antcy++)
    {
      log_file_ants << "Ant cycle : " << antcy << std::endl << std::endl;
      for(int ant=0;ant<optim_param.ACO.ants;ant++)
      {
        for(int pnt=0;pnt<PATH_PNT;pnt++)
        {
          log_file_ants << std::setiosflags(std::ios::fixed)
                        << std::setprecision(2)
                        << std::setw(4)
                        << std::left
                        << logantshost[wa * optim_param.ACO.cycles * optim_param.ACO.ants * PATH_PNT +
                                       antcy * optim_param.ACO.ants * PATH_PNT +
                                       ant * PATH_PNT + 
                                       pnt]
                        << "  ";
        }
        log_file_ants << std::endl;
      }
    }
  }
  
  cudaMemcpy(logphhost,logphdev,3*optim_param.ACO.cycles*sizeof(float)*optim_param.WOA.whales + optim_param.WOA.whales*PATH_PNT*MAX_RID_CONF*optim_param.ACO.cycles*sizeof(float),cudaMemcpyDeviceToHost);

  std::ofstream log_file_ph;
  log_file_ph.open(path + "logph.txt", std::ios_base::app);

  log_file_ph << "Whale cycle : " << wacyc << std::endl << std::endl;
  for(int wa=0;wa<optim_param.WOA.whales;wa++)
  {
    log_file_ph << "Whale number : " << wa << std::endl << std::endl;
    for(int antcy=0;antcy<optim_param.ACO.cycles;antcy++)
    {
      log_file_ph << "Ant cycle : " << antcy << std::endl << std::endl;
      log_file_ph << "  PH MAX : " << logphhost[wa * (optim_param.ACO.cycles * MAX_RID_CONF * PATH_PNT + optim_param.ACO.cycles * 3 ) +
                                                antcy * (MAX_RID_CONF * PATH_PNT + 3)]
                  << "  PH MIN : " << logphhost[wa * (optim_param.ACO.cycles * MAX_RID_CONF * PATH_PNT + optim_param.ACO.cycles * 3 ) +
                                                antcy * (MAX_RID_CONF * PATH_PNT + 3) + 1]
                  <<"  PH EVAP : " << logphhost[wa * (optim_param.ACO.cycles * MAX_RID_CONF * PATH_PNT + optim_param.ACO.cycles * 3 ) +
                                                antcy * (MAX_RID_CONF * PATH_PNT + 3) + 2]
                  << std::endl;
      for(int pnt=0;pnt<PATH_PNT;pnt++)
      {
        for(int conf=0;conf<MAX_RID_CONF;conf++)
        {
          log_file_ph << std::setiosflags(std::ios::fixed)
                      << std::setprecision(2)
                      << std::setw(4)
                      << std::left
                      << logphhost[wa * (optim_param.ACO.cycles * MAX_RID_CONF * PATH_PNT + optim_param.ACO.cycles * 3 ) +
                                   antcy * (MAX_RID_CONF * PATH_PNT + 3) +
                                   pnt * MAX_RID_CONF + 
                                   conf + 3]
                        << "  ";
        }
        log_file_ph << std::endl;
      }
    }
  }
  
  std::fill_n(logantshost,optim_param.WOA.whales*optim_param.ACO.ants*PATH_PNT*optim_param.ACO.cycles,-1);
  check_mem_ = cudaMemcpy(logantsdev,logantshost,optim_param.WOA.whales*optim_param.ACO.ants*PATH_PNT*optim_param.ACO.cycles*sizeof(int),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  std::fill_n(logphhost,3*optim_param.ACO.cycles*optim_param.WOA.whales + optim_param.WOA.whales*PATH_PNT*MAX_RID_CONF*optim_param.ACO.cycles,-1.0);
  check_mem_ = cudaMemcpy(logphdev,logphhost,3*optim_param.ACO.cycles*sizeof(float)*optim_param.WOA.whales + optim_param.WOA.whales*PATH_PNT*MAX_RID_CONF*optim_param.ACO.cycles*sizeof(float),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
  #undef JOINTS_POS
  #undef JOINTS_TOOL
  #undef MAX_RID_CONF
  #undef PATH_PNT
  #undef Z_AXIS_DISCR
  #undef N_WHALES
  #undef RID_CONF
}

void WACOCuda::ClearDevVariables()
#define N_WHALES        optim_param.WOA.whales
#define JOINTS_TOOL     Kine_param.JOINTS_TOOL
#define JOINTS_POS      Kine_param.JOINTS_POS
#define MAX_RID_CONF    Kine_param.MAX_RID_CONF
#define PATH_PNT        Kine_param.PATH_PNT
{
  
  std::fill_n(host_ants_path,N_WHALES*PATH_PNT,-1);
  std::fill_n(already_checked_coll,N_WHALES,false);
  std::fill_n(host_maxFObjWhales,N_WHALES,-1.0);
  std::fill_n(hostcheckreach,N_WHALES,true);
  
  std::fill_n(toolrob_env_check,N_WHALES*PATH_PNT,false);;
  std::fill_n(toolrob_gripp_check,N_WHALES*PATH_PNT,false);
  std::fill_n(toolrob_posit_check,N_WHALES*PATH_PNT,false);
  std::fill_n(max_conf_graph_host,N_WHALES,0);
  
  check_mem_ = cudaMemcpy(dev_already_checked_coll,already_checked_coll,N_WHALES*sizeof(bool),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devgraph,hostgraphempty,Multi_Graph_Helper(0x0,N_WHALES,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_maxFObjWhales,host_maxFObjWhales,sizeof(float)*N_WHALES,cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devcheckreach,hostcheckreach,N_WHALES*sizeof(bool), cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_ants_path,host_ants_path,sizeof(int)*N_WHALES*PATH_PNT,cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);

	free(heuristichost_);
	cudaFree(heuristicdev_);

	check_mem_ = cudaMemset(soldev,0x0,N_WHALES*PATH_PNT*optim_param.ACO.ants*sizeof(int));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemset(max_conf_graph_dev,0,optim_param.WOA.whales*sizeof(int));
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  cudaDeviceSynchronize();
  
  
  
#undef N_WHALES
#undef JOINTS_TOOL
#undef JOINTS_POS
#undef MAX_RID_CONF
#undef PATH_PNT
}

void WACOCuda::RunWACO()
#define N_WHALES        optim_param.WOA.whales
#define JOINTS_TOOL     Kine_param.JOINTS_TOOL
#define JOINTS_POS      Kine_param.JOINTS_POS
#define PATH_PNT        Kine_param.PATH_PNT
#define Z_AXIS_DISCR    Kine_param.Z_AXIS_DISCR
#define RID_CONF        Kine_param.RID_CONF
{
  clock_t time;
  int tot_w_=0;
  double time_cycle[optim_param.WOA.cycles];
  
#if defined(DEBUG_HEUR) || defined(DEBUG_HEUR_POS)
  bool printed = false;
  std::string path_src = ros::package::getPath("itia_ros_wacocuda");
  boost::filesystem::path p_root(path_src);
  boost::filesystem::path folder = p_root.parent_path().parent_path();
  folder /= "logs/";

  strdate = oss.str() + "/";
  folder /= strdate;
  if(!printed){std::cout << "LOG FOLDER : : " << folder.c_str() << std::endl; printed=true;}
  
  if(!boost::filesystem::is_directory(folder)) boost::filesystem::create_directories(folder);
#endif
  
  
  for (int w_iter=0; w_iter<optim_param.WOA.cycles; w_iter++ )
  {
    printf("-----------------WOA ITER %d---------------------\n",w_iter);

    time=clock();
    
    size_t woashared = 3*sizeof(int) + N_WHALES * sizeof(Eigen::Matrix4f);
    
    WOACycle<<<N_WHALES,optim_param.ACO.ants,woashared>>>(dev_robot_tool_lim_joint,dev_robot_pos_lim_joint,w_iter,dev_pathrel,dev_timevector,RobotsTrans,gripper_trans,devgraph,
                             devjointpos,devcheckreach,Kine_param,dev_inv_tool_trans,dev_Robot_Gripper_tcp,max_conf_graph_dev,solIK,hal1_dev,hal2_dev,hal3_dev);
    
    
    cudaError_t err0w = cudaPeekAtLastError();
    cudaError_t err1w = cudaDeviceSynchronize();
    
    if (err0w != cudaSuccess | err1w != cudaSuccess )
    {
      printf("Cuda WOA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err1w));
      printf("WOA peeked error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err0w));
      return;
    }
    
    check_mem_ = cudaMemcpy(max_conf_graph_host,max_conf_graph_dev,N_WHALES * sizeof(int),cudaMemcpyDeviceToHost);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    
    Kine_param.MAX_RID_CONF_REAL = *(std::max_element(max_conf_graph_host,max_conf_graph_host + N_WHALES));
    
    this->AllocMem(N_WHALES,PATH_PNT,Kine_param.MAX_RID_CONF_REAL);
    
    cudaError_t err0a = cudaPeekAtLastError();
    cudaError_t err1a = cudaDeviceSynchronize();
    
    if (err0a != cudaSuccess | err1a != cudaSuccess )
    {
      printf("Cuda Alloc failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err0a));
      printf("Alloc peeked error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err1a));
      return;
    }
      
    if (!heuristic_online)
    {
      
      AntHeuristic<<<N_WHALES,optim_param.ACO.ants>>>(heuristicdev_,devgraph,devcheckreach,optim_param.ACO.beta,dev_robot_tool_speed_joint,
                                                      Kine_param,dev_timevector,dev_multiturn_axis);
      
      cudaError_t err0a = cudaPeekAtLastError();
      cudaError_t err1a = cudaDeviceSynchronize();
      
      if (err0a != cudaSuccess | err1a != cudaSuccess )
      {
        printf("Cuda HEURISTIC failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err0a));
        printf("HEURISTIC peeked error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err1a));
        return;
      }
    }
    
    check_mem_ = cudaMemcpy(host_pos_tcp,dev_pos_tcp,N_WHALES*sizeof(Eigen::Matrix4f),cudaMemcpyDeviceToHost);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);    
    check_mem_ = cudaMemcpy(hostcheckreach,devcheckreach,N_WHALES*sizeof(bool),cudaMemcpyDeviceToHost);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    check_mem_ = cudaMemcpy(hostjointpos,devjointpos,Multi_Positioner_Helper(0x0,N_WHALES,JOINTS_POS).getsizeMulti(),cudaMemcpyDeviceToHost);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    check_mem_ = cudaDeviceSynchronize();
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    
#ifdef DEBUG_HEUR
    check_mem_ = cudaMemcpy(hostgraph,devgraph,Multi_Graph_Helper(0x0,N_WHALES,PATH_PNT,Kine_param.MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyDeviceToHost);
    if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
    
    Multi_Graph_Helper(hostgraph,N_WHALES,PATH_PNT,Kine_param.MAX_RID_CONF,JOINTS_TOOL).dump_log(folder.c_str(),N_WHALES,w_iter);
#endif
      
#ifdef DEBUG_HEUR_POS
      Multi_Positioner_Helper(hostjointpos,N_WHALES,JOINTS_POS).dump_log(folder.c_str(),N_WHALES,w_iter);
#endif
      
#ifdef DEBUG_HEUR_GRAPH
      check_mem_ = cudaMemcpy(heuristichost_,heuristicdev_,Multi_Heur_Helper(0x0,N_WHALES,PATH_PNT,Kine_param.MAX_RID_CONF_REAL).getsizeMulti(),cudaMemcpyDeviceToHost);
      if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
      Multi_Heur_Helper(heuristichost_,N_WHALES,PATH_PNT,Kine_param.MAX_RID_CONF_REAL).dump_log(folder.c_str(),N_WHALES,w_iter);
#endif

    tot_w_=0;
    for(int jp=0;jp<N_WHALES;jp++) tot_w_ += hostcheckreach[jp];
    printf("total solution : %d \n",tot_w_);
    
    if(checkSolNotEmpty(hostcheckreach,N_WHALES))//print e controlla
    {
      printf("-----------------ACO---------------------\n");
      do
      {
        if (!heuristic_online)
        {
          AntCycle<<<N_WHALES,optim_param.ACO.ants,optim_param.ACO.shr_bytes_ACO>>>
                      (devgraph,dev_ants_path,soldev,dev_maxFObjWhales,heuristicdev_,devcheckreach,optim_param.ACO,
                       dev_already_checked_coll,Kine_param, dev_multiturn_axis,logantsdev,logphdev, dev_timevector, 
                       dev_robot_tool_lim_joint,dev_real_joint_angle);
        
          cudaError_t err1 = cudaPeekAtLastError();
          cudaError_t err2 = cudaDeviceSynchronize();
        
          if (err2 != cudaSuccess | err1 != cudaSuccess)
          {
            printf("Cuda ANT Offline failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err2));
            printf("ANT Offline peeked error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err1));
            return;
          } 
        }
        else
        {
          AntCycle_Heur_online<<<N_WHALES,optim_param.ACO.ants,optim_param.ACO.shr_bytes_ACO>>>
                      ( devgraph,dev_ants_path,soldev,dev_maxFObjWhales,heuristicdev_,devcheckreach,optim_param.ACO,
                        dev_already_checked_coll,Kine_param, dev_multiturn_axis,logantsdev,logphdev, dev_timevector, 
                        dev_robot_tool_lim_joint,dev_real_joint_angle, dev_robot_tool_speed_joint, optim_param.ACO.beta);
          
          cudaError_t err1 = cudaPeekAtLastError();
          cudaError_t err2 = cudaDeviceSynchronize();
        
          if (err2 != cudaSuccess | err1 != cudaSuccess)
          {
            printf("Cuda ANT Online failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err2));
            printf("ANT Online peeked error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err1));
            return;
          }
        }
        
        
#ifdef DEBUG_HEUR_ANTS
          this->LogAnts(folder.c_str(),w_iter);
#endif
          
        check_mem_ = cudaMemcpy(hostcheckreach,devcheckreach,N_WHALES*sizeof(bool),cudaMemcpyDeviceToHost);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(host_ants_path,dev_ants_path,PATH_PNT*N_WHALES*sizeof(int),cudaMemcpyDeviceToHost);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(host_maxFObjWhales,dev_maxFObjWhales,N_WHALES*sizeof(float),cudaMemcpyDeviceToHost);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(hostgraph,devgraph,Multi_Graph_Helper(0x0,N_WHALES,PATH_PNT,Kine_param.MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyDeviceToHost);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(already_checked_coll,dev_already_checked_coll,N_WHALES*sizeof(bool),cudaMemcpyDeviceToHost);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(hostBestscore,deviceBestscore,sizeof(float),cudaMemcpyDeviceToHost);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaDeviceSynchronize();
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        
#ifdef DEBUG_HEUR
        for(int jp=0;jp<N_WHALES;jp++) printf("checkreach after ants : %d \n",hostcheckreach[jp]);
#endif
        
        this->WOABest_precheck();
        
        this->CollisionCheck_afterACO(); 
        
        check_mem_ = cudaDeviceSynchronize();
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(devgraph,hostgraph,Multi_Graph_Helper(0x0,N_WHALES,PATH_PNT,Kine_param.MAX_RID_CONF,JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyHostToDevice);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(devcheckreach,hostcheckreach,N_WHALES*sizeof(bool),cudaMemcpyHostToDevice);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(dev_maxFObjWhales,host_maxFObjWhales,N_WHALES*sizeof(float),cudaMemcpyHostToDevice);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaMemcpy(dev_already_checked_coll,already_checked_coll,N_WHALES*sizeof(bool),cudaMemcpyHostToDevice);
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
        check_mem_ = cudaDeviceSynchronize();
        if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);

//         printf("\ncollision check %d\n",already_checked_coll[check_whale_coll]);
//         printf("counter_better_paths %d\n",counter_better_paths);
//         printf("check_whale_coll %d\n",check_whale_coll);
//         printf("no collisions %d\n",no_collisions);
      
      } while (no_collisions==false && counter_better_paths>0 && check_whale_coll>=0);
    }
    else
    {
      check_whale_coll=-1;
    }
    
    WOABestAndHeuristic<<<1,N_WHALES>>>(devicejointbest,devjointpos,w_iter,optim_param.WOA,dev_maxFObjWhales,deviceBestscore,
                                        dev_ants_path,devgraph,devsoltraj,Kine_param,dev_robot_pos_lim_joint, 
                                        check_whale_coll);
    
    cudaError_t err1wo = cudaPeekAtLastError();
    cudaError_t err2wo = cudaDeviceSynchronize();
    
    if (err1wo != cudaSuccess | err2wo != cudaSuccess)
    {
      printf("Cuda WOABestAndHeuristic failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err1wo));
      printf("WOABestAndHeuristic peeked error %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err2wo));
      return;
    }
    
    ClearDevVariables();
  }
  
#undef N_WHALES
#undef JOINTS_TOOL
#undef JOINTS_POS
#undef PATH_PNT
#undef Z_AXIS_DISCR
#undef RID_CONF
}

void WACOCuda::WOABest_precheck()
#define JOINTS_POS   Kine_param.JOINTS_POS
#define JOINTS_TOOL  Kine_param.JOINTS_TOOL
#define N_WHALES     optim_param.WOA.whales
{
  counter_better_paths=0;
  
  for (int i=0;i<N_WHALES;i++)
  {
    if (host_maxFObjWhales[i]>hostBestscore[0] && host_maxFObjWhales[i]>0)
    {
      counter_better_paths++;
    }
    else
    {
      already_checked_coll[i]=true;
    }
  }
  
  float* WOAcyclescorebest = std::max_element(host_maxFObjWhales,host_maxFObjWhales+N_WHALES);
  std::ptrdiff_t best_whale = std::distance(host_maxFObjWhales, WOAcyclescorebest);
  
  std::cout << "best so far:\t" << hostBestscore[0]<< "\tpath migliori best so far:\t" << counter_better_paths <<"\tBest value:\t"<< host_maxFObjWhales[best_whale] << "\n\n";
  
//   for (int pnt=0;pnt<Kine_param.PATH_PNT; pnt++)
//   {
//     printf("%+d\t", host_ants_path[best_whale*Kine_param.PATH_PNT+pnt]);
//   }
//   printf("\n\n");
  
  if (WOAcyclescorebest[0] > hostBestscore[0] && WOAcyclescorebest[0]>0)
  {
    check_whale_coll=best_whale;
  }
  else
  {
    check_whale_coll=-1;
  }
  
#undef JOINTS_POS
#undef JOINTS_TOOL
#undef N_WHALES
}

/////MACRO/////////////////////////////

void WACOCuda::Copy_after_AntHeu()
{
  check_mem_ = cudaMemcpy(hostgraph,devgraph,Multi_Graph_Helper(0x0,optim_param.WOA.whales,Kine_param.PATH_PNT,Kine_param.MAX_RID_CONF,Kine_param.JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(host_pos_tcp,dev_pos_tcp,optim_param.WOA.whales*sizeof(Eigen::Matrix4f),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(hostcheckreach,devcheckreach,optim_param.WOA.whales*sizeof(bool),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(hostjointpos,devjointpos,Multi_Positioner_Helper(0x0,optim_param.WOA.whales,Kine_param.JOINTS_POS).getsizeMulti(),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaDeviceSynchronize();
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
}

void WACOCuda::Copy_after_ACO()
{
  
  check_mem_ = cudaMemcpy(hostcheckreach,devcheckreach,optim_param.WOA.whales*sizeof(bool),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(host_ants_path,dev_ants_path,Kine_param.PATH_PNT*optim_param.WOA.whales*sizeof(int),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(host_maxFObjWhales,dev_maxFObjWhales,optim_param.WOA.whales*sizeof(float),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(hostgraph,devgraph,Multi_Graph_Helper(0x0,optim_param.WOA.whales,Kine_param.PATH_PNT,Kine_param.MAX_RID_CONF,Kine_param.JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(already_checked_coll,dev_already_checked_coll,optim_param.WOA.whales*sizeof(bool),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(hostBestscore,deviceBestscore,sizeof(float),cudaMemcpyDeviceToHost);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaDeviceSynchronize();
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  
}

void WACOCuda::Copy_after_CollisionCheck()
{
  check_mem_ = cudaDeviceSynchronize();
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devgraph,hostgraph,Multi_Graph_Helper(0x0,optim_param.WOA.whales,Kine_param.PATH_PNT,Kine_param.MAX_RID_CONF,Kine_param.JOINTS_TOOL).getSizeMultiGraph(),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(devcheckreach,hostcheckreach,optim_param.WOA.whales*sizeof(bool),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_maxFObjWhales,host_maxFObjWhales,optim_param.WOA.whales*sizeof(float),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaMemcpy(dev_already_checked_coll,already_checked_coll,optim_param.WOA.whales*sizeof(bool),cudaMemcpyHostToDevice);
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
  check_mem_ = cudaDeviceSynchronize();
  if (check_mem_ != cudaSuccess ) HANDLE_ERROR(check_mem_);
}

/////MAIN//////////////////////////////
 
int main(int argc, char *argv[])
{
  ros::init(argc, argv, "itia_ros_wacocuda");
  ros::NodeHandle nh;
  std::string xml_string_tool,xml_string_pos;
  std::string output_folder;
  Kine_const kine_param;
  
  //////////URDF PARSING
  
  if (nh.hasParam("robot_description_positioner") && nh.hasParam("robot_description_tool"))
  {
    nh.getParam("robot_description_positioner",xml_string_pos);
    nh.getParam("robot_description_tool",xml_string_tool);
  }
  else
  {
    ROS_ERROR("No robot robot_description_tool or robot_description_positioner found in parameter server \n");
    return -1;
  }
  
  if (nh.hasParam("output_name") )
  {
    nh.getParam("output_name",output_folder);
  }
  else
  {
    ROS_ERROR("No output folder specified \n");
    return -1;
  }

  urdf::Model urdf_pos,urdf_tool;
  urdf_pos.initString(xml_string_pos);
  urdf_tool.initString(xml_string_tool);
  
  std::map<std::string,urdf::JointSharedPtr>::iterator p;
  std::vector<float> limitlist_pos,limitlist_tool;
  std::vector<std::string> PosJointsNames;
  std::vector<std::string> RobJointsNames;
  
  int dof_pos  = 0;
  int dof_tool = 0;
  
  for(p = urdf_pos.joints_.begin(); p != urdf_pos.joints_.end(); p++) 
  {
    if(p->second->limits != 0 && p->second->type !=0 && p->second->type != 6)
    {
      dof_pos++;
      limitlist_pos.push_back(p->second->limits->lower);
      limitlist_pos.push_back(p->second->limits->upper);
      limitlist_pos.push_back(p->second->limits->velocity);
      PosJointsNames.push_back(p->first);
    }
  }
  
  for(p = urdf_tool.joints_.begin(); p != urdf_tool.joints_.end(); p++) 
  {
    if(p->second->limits != 0 && p->second->type !=0 && p->second->type != 6)
    {
      dof_tool++;
      limitlist_tool.push_back(p->second->limits->lower);
      limitlist_tool.push_back(p->second->limits->upper);
      limitlist_tool.push_back(p->second->limits->velocity);
      RobJointsNames.push_back(p->first);
    }
  }
  
  if(dof_pos == 0 || dof_tool == 0){ ROS_ERROR("No joints in URDF\n"); return -1;}

  kine_param.JOINTS_POS  = dof_pos;
  kine_param.JOINTS_TOOL = dof_tool;

  std::vector<std::string> urdf_paths;
  urdf_paths.resize(5);
  
  if (!nh.getParam("urdf_pos_coarse_coll",urdf_paths[0]))
  {
    ROS_ERROR("Impossible to find urdf_pos_coarse_coll\n");
    return -1;
  }
  
  if (!nh.getParam("urdf_rob_coarse_coll",urdf_paths[1]))
  {
    ROS_ERROR("Impossible to find urdf_rob_coarse_coll\n");
    return -1;
  }
  
  if (!nh.getParam("urdf_env_coarse_coll",urdf_paths[2]))
  {
    ROS_ERROR("Impossible to find urdf_env_coarse_coll\n");
    return -1;
  }
  
  if (!nh.getParam("urdf_pos_fine_coll",urdf_paths[3]))
  {
    ROS_ERROR("Impossible to find urdf_pos_fine_coll\n");
    return -1;
  }
  
  if (!nh.getParam("urdf_rob_fine_coll",urdf_paths[4]))
  {
    ROS_ERROR("Impossible to find urdf_rob_fine_coll\n");
    return -1;
  }
  
  //////////PARAMETERS PARSING
  
  int* multi_turn = new int[dof_tool];// = {0,0,0,1,0,1}; 
  std::vector<int> multi_turn_vec;
  
  if (!nh.getParam("waco_params/UPPER_RANGE",kine_param.UPPER_RANGE))
  {
    ROS_ERROR("Impossible to find waco_params/UPPER_RANGE\n");
    return -1;
  }
  
  if (!nh.getParam("waco_params/LOWER_RANGE",kine_param.LOWER_RANGE))
  {
    ROS_ERROR("Impossible to find waco_params/LOWER_RANGE\n");
    return -1;
  }
  
  if (!nh.getParam("waco_params/STEP",kine_param.STEP))
  {
    ROS_ERROR("Impossible to find waco_params/STEP\n");
    return -1;
  }
  
  if (!nh.getParam("waco_params/multiturn",multi_turn_vec))
  {
    ROS_ERROR("Impossible to find waco_params/multi_turn_vec\n");
    return -1;
  }
  
  std::copy(multi_turn_vec.begin(), multi_turn_vec.end(), multi_turn);
  
  if (!nh.getParam("waco_params/MAX_OPEN_CONE_ANGLE",kine_param.MAX_OPEN_CONE_ANGLE))
  {
    ROS_ERROR("Impossible to find waco_params/MAX_OPEN_CONE_ANGLE\n");
    return -1;
  }
  
  if (!nh.getParam("waco_params/N_OPEN_CONE",kine_param.N_OPEN_CONE))
  {
    ROS_ERROR("Impossible to find waco_params/N_OPEN_CONE\n");
    return -1;
  }
  
  if (!nh.getParam("waco_params/N_ROT_CONE",kine_param.N_ROT_CONE))
  {
    ROS_ERROR("Impossible to find waco_params/N_ROT_CONE\n");
    return -1;
  }
  
  //////////Robots limits setting
  boundaries positioner(kine_param.JOINTS_POS);
  for(int posj=0;posj<kine_param.JOINTS_POS;posj++)
  {
    positioner.pos_joint[posj].lower = limitlist_pos.at(3*posj);
    positioner.pos_joint[posj].upper = limitlist_pos.at(3*posj + 1);
  }
  
  boundaries robot_tool(kine_param.JOINTS_TOOL,multi_turn);
  for(int robj=0;robj<kine_param.JOINTS_POS;robj++)
  {
    robot_tool.pos_joint[robj].lower   = limitlist_tool.at(3*robj);
    robot_tool.pos_joint[robj].upper   = limitlist_tool.at(3*robj + 1);
    robot_tool.speed_joint[robj].upper = limitlist_tool.at(3*robj + 2);
  }
  
#if defined(DEBUG_HEUR) || defined(DEBUG_HEUR_POS)

    std::cout << "------------------------------DEBUG--------------------------------" << std::endl;
    std::cout << "------------------------------DEBUG--------------------------------" << std::endl;
    
    std::cout << "positioner limits : " << std::endl;
    for(int jn=0;jn<dof_pos;jn++)
    {
      std::cout << PosJointsNames[jn] << "  :" << positioner.pos_joint[jn].lower << "  " << positioner.pos_joint[jn].upper << "  " << std::endl;
    }
    std::cout << "positioner joints : " << dof_pos << std::endl;
    
    std::cout << "tool robot limits : " << std::endl;
    for(int jn=0;jn<dof_tool;jn++)
    {
      std::cout << RobJointsNames[jn] << "  :" << robot_tool.pos_joint[jn].lower << "  " << robot_tool.pos_joint[jn].upper << "  " << robot_tool.speed_joint[jn].upper << std::endl;
    }
    std::cout << "robot joints : " << dof_tool << std::endl;
    
#endif

  //////////KINEMATIC REDUNDANCY PARAMETERS
  float range = (kine_param.UPPER_RANGE-kine_param.LOWER_RANGE);
  
  if(range >= ((2 * PI_F) - kine_param.STEP )) 
  {
    kine_param.Z_AXIS_DISCR = (int) ((kine_param.UPPER_RANGE-kine_param.LOWER_RANGE)/kine_param.STEP);
  }
  else
  {
    kine_param.Z_AXIS_DISCR = (int) ((kine_param.UPPER_RANGE-kine_param.LOWER_RANGE)/kine_param.STEP) + 1 ;
  }
  
  if (kine_param.N_ROT_CONE > 1)
  {
    kine_param.STEP_ROT_CONE=2*PI_F/kine_param.N_ROT_CONE;
  }
  else
  {
    kine_param.STEP_ROT_CONE = 0;
  }
  
  if (kine_param.N_OPEN_CONE>1)
  {
    kine_param.STEP_OPEN_CONE_ANGLE=kine_param.MAX_OPEN_CONE_ANGLE/(kine_param.N_OPEN_CONE-1);
  }
  else
  {
    kine_param.STEP_OPEN_CONE_ANGLE=0;
  }
  


  kine_param.RID_CONE=kine_param.N_OPEN_CONE*kine_param.N_ROT_CONE;
  kine_param.MAX_RID_CONF = kine_param.Z_AXIS_DISCR*8*kine_param.RID_CONE;
  std::cout << "MAX_RID_CONF : " << kine_param.MAX_RID_CONF << std::endl
            << "UPPER_RANGE  : " << kine_param.UPPER_RANGE << std::endl
            << "LOWER_RANGE  : " << kine_param.LOWER_RANGE << std::endl
            << "STEP  : " << kine_param.STEP << std::endl
            << "Z_AXIS_DISCR  : " << kine_param.Z_AXIS_DISCR << std::endl
            << "RID_CONE " << kine_param.RID_CONE << std::endl
            << std::endl << std::endl ;
  kine_param.RID_CONF=kine_param.Z_AXIS_DISCR*kine_param.RID_CONE;
  
  //////////TRANSOFMATION MATRICES
  
  Eigen::Matrix4f RTOOL_RPOS;
  std::vector<float> RTOOL_RPOS_vec(16);
  
  if (!nh.getParam("waco_params/RTOOL_RPOS",RTOOL_RPOS_vec))
  {
    ROS_ERROR("Impossible to find waco_params/RTOOL_RPOS\n");
    return -1;
  }
  
  RTOOL_RPOS(0,0)=RTOOL_RPOS_vec[0] ; RTOOL_RPOS(0,1)=RTOOL_RPOS_vec[1] ; RTOOL_RPOS(0,2)=RTOOL_RPOS_vec[2] ; RTOOL_RPOS(0,3)=RTOOL_RPOS_vec[3];
  RTOOL_RPOS(1,0)=RTOOL_RPOS_vec[4] ; RTOOL_RPOS(1,1)=RTOOL_RPOS_vec[5] ; RTOOL_RPOS(1,2)=RTOOL_RPOS_vec[6] ; RTOOL_RPOS(1,3)=RTOOL_RPOS_vec[7];
  RTOOL_RPOS(2,0)=RTOOL_RPOS_vec[8] ; RTOOL_RPOS(2,1)=RTOOL_RPOS_vec[9] ; RTOOL_RPOS(2,2)=RTOOL_RPOS_vec[10]; RTOOL_RPOS(2,3)=RTOOL_RPOS_vec[11];
  RTOOL_RPOS(3,0)=RTOOL_RPOS_vec[12]; RTOOL_RPOS(3,1)=RTOOL_RPOS_vec[13]; RTOOL_RPOS(3,2)=RTOOL_RPOS_vec[14]; RTOOL_RPOS(3,3)=RTOOL_RPOS_vec[15];

  std::cout << "RELATIVE TRANSOFMATION : \n " << RTOOL_RPOS << std::endl;
  
  Eigen::Matrix4f GRIPPER;
  std::vector<float> GRIPPER_vec(16);
  if (!nh.getParam("waco_params/GRIPPER",GRIPPER_vec))
  {
    ROS_ERROR("Impossible to find waco_params/GRIPPER\n");
    return -1;
  }
  
  GRIPPER(0,0) =GRIPPER_vec[0] ;GRIPPER(0,1)=GRIPPER_vec[1] ;GRIPPER(0,2)=GRIPPER_vec[2] ;GRIPPER(0,3)=GRIPPER_vec[3];
  GRIPPER(1,0) =GRIPPER_vec[4] ;GRIPPER(1,1)=GRIPPER_vec[5] ;GRIPPER(1,2)=GRIPPER_vec[6] ;GRIPPER(1,3)=GRIPPER_vec[7];
  GRIPPER(2,0) =GRIPPER_vec[8] ;GRIPPER(2,1)=GRIPPER_vec[9] ;GRIPPER(2,2)=GRIPPER_vec[10];GRIPPER(2,3)=GRIPPER_vec[11];
  GRIPPER(3,0) =GRIPPER_vec[12];GRIPPER(3,1)=GRIPPER_vec[13];GRIPPER(3,2)=GRIPPER_vec[14];GRIPPER(3,3)=GRIPPER_vec[15];

  std::cout << "GRIPPER : \n " << GRIPPER << std::endl;
  
  Eigen::Matrix4f TOOL;  
  std::vector<float> TOOL_vec(16);  
  if (!nh.getParam("waco_params/TOOL",TOOL_vec))
  {
    ROS_ERROR("Impossible to find waco_params/TOOL\n");
    return -1;
  }
  
  TOOL(0,0) =TOOL_vec[0] ;TOOL(0,1)=TOOL_vec[1] ;TOOL(0,2)=TOOL_vec[2] ;TOOL(0,3)=TOOL_vec[3];
  TOOL(1,0) =TOOL_vec[4] ;TOOL(1,1)=TOOL_vec[5] ;TOOL(1,2)=TOOL_vec[6] ;TOOL(1,3)=TOOL_vec[7];
  TOOL(2,0) =TOOL_vec[8] ;TOOL(2,1)=TOOL_vec[9] ;TOOL(2,2)=TOOL_vec[10];TOOL(2,3)=TOOL_vec[11];
  TOOL(3,0) =TOOL_vec[12];TOOL(3,1)=TOOL_vec[13];TOOL(3,2)=TOOL_vec[14];TOOL(3,3)=TOOL_vec[15];

  std::cout << "TOOL :\n " << TOOL << std::endl;
  
  //////////TRAJECTORY PARSING
  bool traj_from_csv;
  if (!nh.getParam("traj_from_csv",traj_from_csv))
  {
    ROS_ERROR("Impossible to find traj_from_csv\n");
    return -1;
  }

  std::string path_csv;
  std::string path_shape;
  int pointsnumber;
  float radius;
  float speed_tcp;
  
  if(traj_from_csv)
  { 
    if (!nh.getParam("path_csv",path_csv))
    {
      ROS_ERROR("Impossible to find path_csv\n");
      return -1;
    }
    ROS_INFO("Trajectory loaded from csv file \n");
  }
  else
  {
    if (!nh.getParam("traj_params/path_shape",path_shape))
    {
      ROS_ERROR("Impossible to find path_shape\n");
      return -1;
    }
    
    if (!nh.getParam("traj_params/pointsnumber",pointsnumber))
    {
      ROS_ERROR("Impossible to find pointsnumber\n");
      return -1;
    }
    
    if (!nh.getParam("traj_params/radius",radius))
    {
      ROS_ERROR("Impossible to find radius\n");
      return -1;
    }
    
    if (!nh.getParam("traj_params/speed",speed_tcp))
    {
      ROS_ERROR("Impossible to find speed\n");
      return -1;
    }
    ROS_INFO("Trajectory %s with %d points \n",path_shape.c_str(),pointsnumber);
  }
  
  trajectory path = traj_from_csv ? trajectory(path_csv) : trajectory(path_shape, radius, pointsnumber, speed_tcp);
  
// #if defined(DEBUG_HEUR) || defined(DEBUG_HEUR_POS)
//   for(int pnt=0;pnt<pointsnumber;pnt++) std::cout << path.points[pnt] << std::endl;
// #endif
  
  kine_param.PATH_PNT = traj_from_csv ? path.get_points_num() : pointsnumber;
  
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int multiproc  = deviceProp.multiProcessorCount;
  float memtot   = (float)deviceProp.totalGlobalMem/1048576.0f;
  int maxthreads = deviceProp.maxThreadsPerBlock;
  
  int whale_num ;
  int whale_cycles ;
  float ALim ;
  int ants = maxthreads;
  int ant_cycles ;
  float beta;
  
  if (!nh.getParam("HEURISTIC/whale_num",whale_num))
  {
    ROS_ERROR("Impossible to find whale_num\n");
    return -1;
  }
  
  if (!nh.getParam("HEURISTIC/whale_cycles",whale_cycles))
  {
    ROS_ERROR("Impossible to find whale_cycles\n");
    return -1;
  }
  
  if (!nh.getParam("HEURISTIC/ALim",ALim))
  {
    ROS_ERROR("Impossible to find ALim\n");
    return -1;
  }
  
  if (!nh.getParam("HEURISTIC/ant_cycles",ant_cycles))
  {
    ROS_ERROR("Impossible to find ant_cycles\n");
    return -1;
  }
  
  if (!nh.getParam("HEURISTIC/beta",beta))
  {
    ROS_ERROR("Impossible to find beta\n");
    return -1;
  }
  
  if((ants & (ants - 1))!=0)
  {
    ROS_ERROR("wrong number of ants\n");
    return -1;
  }

  WACOCuda WACO(whale_num,whale_cycles,positioner,robot_tool,ALim,&path,RTOOL_RPOS,GRIPPER,ants,ant_cycles,&kine_param,TOOL,beta,PosJointsNames,RobJointsNames,urdf_paths);

  clock_t t0=clock();

  WACO.RunWACO();

  double t_WACO= (double) (clock()-t0)/CLOCKS_PER_SEC;

  WACO.Copytohost();
  
  
  
  std::string folder_params= "/"+std::to_string(whale_num)+"_"+ std::to_string(whale_cycles)+"_"+ std::to_string(ants)+"_"+ std::to_string(ant_cycles);
  std::string path_src = ros::package::getPath("itia_ros_wacocuda");
  
  if (!boost::filesystem::exists(path_src + "/output/" + output_folder))
    boost::filesystem::create_directory(path_src + "/output/" + output_folder);
  
  std::string strdate;
  std::ostringstream oss;
  
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
  
  std::string folder_path=path_src + "/output/" + output_folder + folder_params + "/" +  oss.str(); 

  save_data print_on_file(folder_path);
  std::cout << "saving file in " << folder_path << std::endl;
  print_on_file.print_csv_JointPositioner("/positioner.txt",kine_param.JOINTS_POS,WACO.hostjointbest);
  print_on_file.print_csv_JointRobot("/Robot.txt",kine_param.PATH_PNT,kine_param.JOINTS_TOOL,WACO.hostsoltraj);
  print_on_file.print_csv_Float("/score.txt",WACO.hostBestscore[0]);
  print_on_file.print_csv_Float("/execution_time.txt",t_WACO);
  print_on_file.print_csv_fobj_history("/fobj_history.txt", WACO.fobj_history, whale_cycles, whale_num);

  std::cout << "Computation time " << t_WACO << std::endl; 
  
  std::cout << "file saved in\n" << folder_path << "\n";

  //test.~WACOCuda();

  return 0;

}
