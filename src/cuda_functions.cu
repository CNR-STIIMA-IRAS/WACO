#include <cuda_functions.h>

//////////// CUDA FUNCTIONS

__device__ int nosolik;
__device__ int validsol;

__device__ __forceinline__ void Obj_fun_jointspeed_jointlimits
                                    ( int* solptr,float* phobjpnt,Multi_Graph_Helper& Graph, Kine_const kine_param, int* multiturn_axis, 
                                      float *timevector, joint_boundary *joint_limits, float* real_joint_angle  )
{
  float phi_speed = 0;
  float phi_jointlimits=0;
  float tot   = 0;
  float tot_speed=0;
  float tot_limits=0;
  int ant = threadIdx.x;
  int whale = blockIdx.x;
  int node_conf, next_node_conf;
  bool feasible=true;
  
  for (int jnt=0; jnt<kine_param.JOINTS_TOOL; jnt++)
  {
    real_joint_angle[jnt]=Graph(whale,0,solptr[ant*kine_param.PATH_PNT],jnt);
  }
    
  for(int pnt=0;pnt<kine_param.PATH_PNT-1;pnt++)
  {
    node_conf=solptr[ant*kine_param.PATH_PNT + pnt];
    next_node_conf=solptr[ant*kine_param.PATH_PNT + pnt+1];
    
//     for(int jnt=0;jnt<Graph.ndof_;jnt++)
//     {
//       phi_jointlimits += powf(abs(Graph(whale,pnt,node_conf,jnt)-joint_limits[jnt].middle)/(0.5*(joint_limits[jnt].upper-joint_limits[jnt].lower)), 2);
//     }
    
    if (pnt<kine_param.PATH_PNT-1)
    {
      for(int jnt=0;jnt<Graph.ndof_;jnt++)
      {
        if (!(bool)multiturn_axis[jnt])
        {
          phi_speed += ((Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt,node_conf,jnt))/timevector[pnt])*((Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt,node_conf,jnt))/timevector[pnt]);
        }
        else
        {
          if (abs(Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt,node_conf,jnt))<=PI_F)
          {
            real_joint_angle[jnt] += Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt,node_conf,jnt);
            phi_speed += ((Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt,node_conf,jnt))/timevector[pnt])*((Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt,node_conf,jnt))/timevector[pnt]);       
          }
          else
          {
            if (Graph(whale,pnt+1,next_node_conf,jnt)>0)
            {
              real_joint_angle[jnt] -= 2.0*PI_F-(Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt,node_conf,jnt));
              phi_speed += __powf((2.0*PI_F-(Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt+1,node_conf,jnt)))/timevector[pnt],2);
            }
            else
            {
              real_joint_angle[jnt] += 2.0*PI_F+(Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt,node_conf,jnt));
              phi_speed += __powf((2.0*PI_F+(Graph(whale,pnt+1,next_node_conf,jnt)-Graph(whale,pnt+1,node_conf,jnt)))/timevector[pnt],2);
            }
          }
          if ((real_joint_angle[jnt] > joint_limits[jnt].upper) || (real_joint_angle[jnt] < joint_limits[jnt].lower))
          {
            phobjpnt[0]=0;
            return;
          }
        }
      }
    }
    tot_speed+= phi_speed;
  }
  *phobjpnt=__fdividef(kine_param.PATH_PNT ,tot_speed);
}


__device__ __forceinline__ void EvalDist(int* solptr,float* phobjpnt,Multi_Graph_Helper& Graph,Kine_const kine_param, int* multiturn_axis)
{
  float phinc = 0;
  float tot   = 0;
  int ant = threadIdx.x;
  int whale = blockIdx.x;
  for(int pnt=0;pnt<kine_param.PATH_PNT-1;pnt++)
  {
    int config=solptr[ant*kine_param.PATH_PNT + pnt];
    for(int jnt=0;jnt<Graph.ndof_;jnt++)
    {
      if (!(bool)multiturn_axis[jnt])
      {
        phinc += powf((Graph(whale,pnt,config,jnt)-Graph(whale,pnt+1,config+1,jnt)),2);
      }
      else
      {
        if ((Graph(whale,pnt,config,jnt)-Graph(whale,pnt+1,config+1,jnt))<=PI_F)
        {
          phinc += powf((Graph(whale,pnt,config,jnt)-Graph(whale,pnt+1,config+1,jnt)),2);       
        }
        else
        {
          phinc += powf(2*PI_F-(Graph(whale,pnt,config,jnt)-Graph(whale,pnt+1,config+1,jnt)),2);               
        }
      }
    }
    tot += phinc;
  }
  *phobjpnt=fdividef(1,tot);
}

__global__ void AntCycle_Heur_online(void* graph,int* solution,int* sol,float* maxfobjtot,void* heur_onl,bool* feasible,ACO_params ACO,
                                    bool* coll_check,Kine_const kine_param, int* multiturn_axis, int* logantsdev,float* logphdev,
                                    float* timevector, joint_boundary *joint_limits,float* real_joint_angle, 
                                    joint_boundary *joint_limits_speed, float beta)
#define PATH_PNT           kine_param.PATH_PNT
#define JOINTS_POS         kine_param.JOINTS_POS
#define JOINTS_TOOL        kine_param.JOINTS_TOOL
#define MAX_RID_CONF       kine_param.MAX_RID_CONF
#define MAX_RID_CONF_REAL  kine_param.MAX_RID_CONF_REAL
{
  
  float rnd_sel = 0.0;
  float prev_ph = 0.0;
  curandState_t state;
  int actsol = 0;
  int whale  = blockIdx.x;
  int n_ants = blockDim.x;
  bool sol_found;
  bool first_set = true;
  
  int current_conf=0;
  float tot=0;
  bool skip=false;
  int neigh   = 0;
  
  int cycth_heur     = ceil(((float)MAX_RID_CONF_REAL/(float)blockDim.x));
  int cycth     = (PATH_PNT/(int)blockDim.x) + 1;
  
  extern __shared__ int shmemant[];
  float     * phobj         = (float  *)&shmemant[0];
  float     * phobjcpy      = (float  *)&phobj[n_ants];
  float     * tot_ph        = (float  *)&phobjcpy[n_ants];
  int       * ind           = (int    *)&tot_ph[n_ants];
  int       * indtot        = (int    *)&ind[1];
  float     * phmin         = (float  *)&indtot[1];
  float     * phmax         = (float  *)&phmin[1];
  int       * hold          = (int    *)&phmax[1];
  int       * holdtot       = (int    *)&hold[1];
  float     * maxfobjcycle  = (float  *)&holdtot[1];
  int       * best          = (int    *)&maxfobjcycle[1];
  int       * bestcyc       = (int    *)&best[1];
  float     *PHevap         = (float  *)&bestcyc[1];
  float     *nodes          = (float  *)&PHevap[1];
  
  maxfobjtot = maxfobjtot + whale;
  solution   = solution   + whale*PATH_PNT;
  sol        = sol        + whale*PATH_PNT*n_ants;
  feasible   = feasible   + whale;
  coll_check = coll_check + whale;
  
  
  Multi_Graph_Helper Graph(graph,gridDim.x,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL);
  real_joint_angle = real_joint_angle + JOINTS_TOOL*blockDim.x*blockIdx.x + JOINTS_TOOL*threadIdx.x;
  Multi_Heur_Helper Heur(heur_onl,gridDim.x,1,MAX_RID_CONF_REAL);
 
  curand_init(clock64() ,threadIdx.x+blockDim.x*whale, clock64(), &state);

  if(threadIdx.x==0)
  {
    ind[0]        = 0;
    indtot[0]     = 0;
    maxfobjtot[0] = -1.0;
    phmax[0]      = 1;
    phmin[0]      = 1;
    PHevap[0]     = 0;
    nodes[0]      = 0.0;
  }
  __syncthreads();
  
  for(int cy=0;cy<cycth;cy++)
  {
    int thcyc = threadIdx.x + cy * blockDim.x;
    if(thcyc<PATH_PNT)
    {
      for(int mm=0;mm<MAX_RID_CONF_REAL;mm++)
      {
        fAtomicAdd(nodes,(float)Graph.getCh(whale,thcyc,mm));
      }
    }
  }
  
  __syncthreads();
  
  if(threadIdx.x==0)
  {
    nodes[0] = nodes[0] / PATH_PNT;
    float num=1.0-pow(0.05,1.0/nodes[0]);
    float den=((nodes[0]*0.5)-1.0)*(pow(0.05,1.0/nodes[0]));
    PHevap[0] = 1.0 - pow((num/den),(1.0/(float)ACO.cycles));
  }

  __syncthreads();
  __threadfence();
  
  if (*feasible && (*coll_check)==false) //TODO formiche pd pd pd
  {
    for(int cyc=0;cyc<ACO.cycles;cyc++)
    {
      if(threadIdx.x==0)
      {
        maxfobjcycle[0] = -1.0;
        hold[0]         = 1;
        holdtot[0]      = 1;
        best[0]         = 0;
        bestcyc[0]      = 0;
      }
      actsol = 0;
      sol_found = true;
      __syncthreads(); 
      __threadfence();
      
      
      for (int pnt=0 ; pnt<PATH_PNT ; pnt++) //PROBABILISTIC SELECTION IMPLEMENTATION
      {
        
        sol[PATH_PNT * threadIdx.x + pnt] = -1;
        tot_ph[threadIdx.x] = 0.0;
        rnd_sel     = 0.0;
        
        for(int q=0;q<cycth_heur;q++)
        {
          current_conf = threadIdx.x + q*blockDim.x;
          if(current_conf<MAX_RID_CONF_REAL)
          {
            for(int next_conf=0;next_conf<MAX_RID_CONF_REAL;next_conf++)
            {
              Heur.setHeuVal(whale,0,current_conf,next_conf,0.0);
              if (pnt==0)
              {
                neigh = 0;
                neigh = Graph.getNeigh(whale,current_conf); 
                Heur.setHeuVal(whale,0,0,current_conf,(float)neigh);
              }
              else
              {
                if( Graph.getCh(whale,pnt-1,current_conf) && Graph.getCh(whale,pnt,next_conf))
                {
                  
                  skip = false;
                  for(int jnt=0;jnt<JOINTS_TOOL;jnt++)
                  {
                    if((fabsf(__fdividef((Graph(whale,pnt-1,current_conf,jnt) - Graph(whale,pnt,next_conf,jnt)),timevector[pnt-1]))) < joint_limits_speed[jnt].upper)
                    {
                      Heur.increaseHeu(whale,0,current_conf,next_conf, fabsf(Graph(whale,pnt-1,current_conf,jnt) - Graph(whale,pnt,next_conf,jnt)));
                    }
                    else
                    {
                      if((bool)multiturn_axis[jnt])
                      {
                        if(Graph(whale,pnt-1,current_conf,jnt)*Graph(whale,pnt,next_conf,jnt)<0 && 
                          (((2*PI_F- fabsf(__fdividef(Graph(whale,pnt-1,current_conf,jnt) - Graph(whale,pnt,next_conf,jnt),timevector[pnt-1])))) < joint_limits_speed[jnt].upper))
                        {
                          Heur.increaseHeu(whale,0,current_conf,next_conf, 2*PI_F- fabsf((Graph(whale,pnt-1,current_conf,jnt) - Graph(whale,pnt,next_conf,jnt))));
                        }
                        else
                        {
                          Heur.setHeuVal(whale,0,current_conf,next_conf,CUDART_NAN_F);
                          skip = true;
                          break;
                        }
                      }
                      else
                      {
                        Heur.setHeuVal(whale,0,current_conf,next_conf,CUDART_NAN_F);
                        skip = true;
                        break;
                      }
                    }
                  }
                  
                  if(skip) continue;
                  
                  if(Heur(whale,0,current_conf,next_conf)!=0 && Heur(whale,0,current_conf,next_conf)!=CUDART_NAN_F)
                  {
                    Heur.setHeuVal(whale,0,current_conf,next_conf,__powf(Heur(whale,0,current_conf,next_conf),(-beta)));
                  }
                }
                else
                {
                  Heur.setHeuVal(whale,0,current_conf,next_conf,CUDART_NAN_F);
                }
              }
            }
          }
        }

        for(int q=0;q<cycth;q++)
        {
          current_conf = threadIdx.x + q*blockDim.x;
          if(current_conf<MAX_RID_CONF_REAL)
          {
            tot = 0;
            for(int next_conf=0;next_conf<MAX_RID_CONF_REAL;next_conf++) 
            {
              if(!isnan(Heur(whale,0,current_conf,next_conf))) tot+=  Heur(whale,0,current_conf,next_conf);
            }
            if(tot!=0)
            {
              for(int next_conf=0;next_conf<MAX_RID_CONF_REAL;next_conf++) 
              {
                Heur.setHeuVal(whale,0,current_conf,next_conf,__fdividef( MAX_RID_CONF_REAL*(Heur(whale,0,current_conf,next_conf)),tot));
              }
            }
          }
        }
        
        __syncthreads();
        __threadfence();
        
        for(int next_conf=0;next_conf<MAX_RID_CONF_REAL;next_conf++)
        {
          tot_ph[threadIdx.x] += Heur(whale,0,actsol,next_conf) * Graph.getPh(whale,pnt,next_conf);
        }    
        
        __syncthreads();
        __threadfence();
        
        if(tot_ph[threadIdx.x] > 0)
        {
          rnd_sel = curand_uniform(&state) * tot_ph[threadIdx.x];
        }
        else
        {
          sol_found = false;
        }
        
        __syncthreads();
        __threadfence();
        
        prev_ph = 0.0;
        
        if(sol_found)
        {
          for(int conf=0;conf<MAX_RID_CONF_REAL;conf++)
          {
            prev_ph += Graph.getPh(whale,pnt,conf) * Heur(whale,0,actsol,conf);
            
            if(rnd_sel <= prev_ph  )
            {
              sol[threadIdx.x*PATH_PNT + pnt] = conf;
              actsol = conf;
    #ifdef DEBUG_HEUR
              logantsdev[whale*n_ants*PATH_PNT*ACO.cycles + cyc*PATH_PNT*ACO.ants + threadIdx.x*PATH_PNT + pnt] = conf;
    #endif
              break;
            }

            if(conf==((MAX_RID_CONF_REAL)-1) && rnd_sel>prev_ph)
            {
              printf("ERROR in ACO %f>%f>%f thread : %d  block : %d cyc :%d conf : %d pnt :%d\n",rnd_sel,prev_ph,tot_ph[threadIdx.x],threadIdx.x,blockIdx.x,cyc,conf,pnt);
              actsol = conf;
              sol[threadIdx.x*PATH_PNT + pnt]=conf;
            }
          }
        }
        __syncthreads();
        __threadfence();
      }
      
      phobj[threadIdx.x]     = 0.0;
      phobjcpy[threadIdx.x]  = 0.0;

      __syncthreads();
      __threadfence();
      
//       if(sol_found) EvalDist(sol,&phobj[threadIdx.x],Graph,kine_param, multiturn_axis);
      if(sol_found) Obj_fun_jointspeed_jointlimits(sol,&phobj[threadIdx.x],Graph,kine_param, multiturn_axis,timevector,joint_limits,real_joint_angle);
//       printf("Obj func : %f    evap : %f    sol found : %d \n",phobj[threadIdx.x],ACO.PHevap,sol_found);
      phobjcpy[threadIdx.x] = phobj[threadIdx.x];
      
      __syncthreads();
      __threadfence();
      
      for (unsigned int s=blockDim.x/2; s>0; s>>=1)
      {
        if (threadIdx.x < s) phobjcpy[threadIdx.x] = max(phobjcpy[threadIdx.x], phobjcpy[threadIdx.x + s]);
        __syncthreads();
      }
      
      if (threadIdx.x == 0 && phobjcpy[0] > 0.0)
      {
        maxfobjcycle[0] = phobjcpy[0];
        bestcyc[0]      = 1;
        
        if(maxfobjtot[0] < maxfobjcycle[0])
        {
          best[0] = 1;
          maxfobjtot[0] = maxfobjcycle[0];
        }
        
        if(first_set)
        {
          phmax[0] = __fdividef(maxfobjtot[0],PHevap[0]);
          phmin[0] = __fdividef(maxfobjtot[0],PHevap[0]);
          first_set = false;
        }  
        else
        {
          phmax[0] = __fdividef(maxfobjtot[0],PHevap[0]);
          phmin[0] = __fdividef((phmax[0]*(1.0-powf(0.05,1.0/nodes[0]))),(((nodes[0]/2.0)-1.0)*(powf(0.05,1.0/nodes[0]))));
        }
  #ifdef DEBUG_HEUR
          logphdev[whale*PATH_PNT*ACO.cycles*MAX_RID_CONF + whale*3*ACO.cycles +
                    cyc*PATH_PNT*MAX_RID_CONF + cyc * 3]     = phmax[0];
          logphdev[whale*PATH_PNT*ACO.cycles*MAX_RID_CONF + whale*3*ACO.cycles +
                    cyc*PATH_PNT*MAX_RID_CONF + cyc * 3 + 1] = phmin[0];
          logphdev[whale*PATH_PNT*ACO.cycles*MAX_RID_CONF + whale*3*ACO.cycles +
                    cyc*PATH_PNT*MAX_RID_CONF + cyc * 3 + 2] = PHevap[0];
  #endif
      }
      
  //     if(threadIdx.x==0) printf("whale : %d  cycle %d maxfobjcycle : %f  maxfobjtot : %f max-min PH : %f %f \n",whale,cyc,maxfobjcycle[0],maxfobjtot[0],phmax[0],phmin[0]);//TODO
      __syncthreads();
      
      //     BEST SOLUTION PICK
      if(maxfobjcycle[0]==phobj[threadIdx.x] && atomicExch(hold,(int)(maxfobjcycle[0]!=phobj[threadIdx.x])) && bestcyc[0])
      { 
        if(!sol_found) printf("NO SOL megapd1 whale : %d thread : %d cyc : %d maxfobjcycle : %f \n\n",whale,threadIdx.x,cyc,maxfobjcycle[0]);
        ind[0]=threadIdx.x;
      }
      
      if(maxfobjtot[0]==phobj[threadIdx.x] && atomicExch(holdtot,(int)(maxfobjtot[0]!=phobj[threadIdx.x])) && best[0])
      {
        if(!sol_found) printf("megapd2????????????????????????????????????????????????????????????????????????????\n\n");
        indtot[0]=threadIdx.x;
      }
      
      __syncthreads();
      __threadfence();
      
      //EVAPORATION
      for(int cy=0;cy<cycth;cy++)
      {
        int thcyc = threadIdx.x + cy * blockDim.x;
        if(thcyc<PATH_PNT)
        {
          for(int mm=0;mm<MAX_RID_CONF_REAL;mm++)
          {
            if(Graph.getCh(whale,thcyc,mm))
            {
              atomicMul(Graph.getPhAdd(whale,thcyc,mm),(1-PHevap[0]));
              if(Graph.getPh(whale,thcyc,mm) < phmin[0]) 
              {
                Graph.setPh(whale,thcyc,mm,phmin[0]);
              }
            }
          }
        }
      }
      
      __syncthreads();
      __threadfence();
      //PHERORMONE INCREASE
      for(int cy=0;cy<cycth;cy++)
      {
        int thcyc = threadIdx.x + cy * blockDim.x;
        if(thcyc<PATH_PNT)
        {
          fAtomicAdd(Graph.getPhAdd(whale,thcyc,sol[thcyc+PATH_PNT*ind[0]]),phobj[ind[0]]);
        }
      }
      
      __syncthreads();
      __threadfence();
      
      //PHERORMONE SATURATION
      for(int cy=0;cy<cycth;cy++)
      {
        int thcyc = threadIdx.x + cy * blockDim.x;
        if(thcyc<PATH_PNT)
        {
          for(int mm=0;mm<MAX_RID_CONF_REAL;mm++)
          {
            if(Graph.getCh(whale,thcyc,mm))
            {
              if(Graph.getPh(whale,thcyc,mm) < phmin[0]) 
              {
                Graph.setPh(whale,thcyc,mm,phmin[0]);
              }
              if(Graph.getPh(whale,thcyc,mm) > phmax[0]) 
              {
                Graph.setPh(whale,thcyc,mm,phmax[0]);
              }
            }
  #ifdef DEBUG_HEUR
            logphdev[whale*PATH_PNT*ACO.cycles*MAX_RID_CONF + whale*3*ACO.cycles +
                    cyc*PATH_PNT*MAX_RID_CONF + cyc * 3 +
                    thcyc*MAX_RID_CONF +
                    mm + 3] = Graph.getPh(whale,thcyc,mm);
  #endif
          }
        }
      }
    
      __syncthreads();
      __threadfence();
      
      //BEST SOLUTION 
      if(best[0]==1)
      {
        for(int cy=0;cy<cycth;cy++)
        {
          int thcyc = threadIdx.x + cy * blockDim.x;
          if(thcyc<PATH_PNT)
          {
            solution[thcyc] = sol[(indtot[0]) * PATH_PNT + thcyc]; 
          }
        }
        __syncthreads();
        __threadfence();
      }
    }
    
    if (threadIdx.x<PATH_PNT)//TODO if = -1 unfeasible??
    {
      if(maxfobjtot[0] < 0.0)
      {
        feasible[0]   = false;
        coll_check[0] = true;
        return;//TODO
      }
      
      if (Graph.getCh(whale,threadIdx.x,solution[threadIdx.x])==false)
      {
        printf("MEGA PDDDDDDD-- -- -- -- -- -- whale : %d \n\n",whale);
        maxfobjtot[0] = -1;
        feasible[0]   = false;
        coll_check[0] = true;
      }
    }
    __syncthreads();
    __threadfence();
  }
  
#undef PATH_PNT
#undef JOINTS_TOOL
#undef JOINTS_POS
#undef MAX_RID_CONF
#undef MAX_RID_CONF_REAL
}


__global__ void AntHeuristic(void* heur,void* graph_ptr, bool* feasible, float beta,joint_boundary* robot2_speed_lim,Kine_const kine_param,
                             float* timevector, int* multiturn_axis)
#define PATH_PNT           kine_param.PATH_PNT
#define JOINTS_TOOL        kine_param.JOINTS_TOOL
#define JOINTS_POS         kine_param.JOINTS_POS
#define MAX_RID_CONF       kine_param.MAX_RID_CONF
#define MAX_RID_CONF_REAL  kine_param.MAX_RID_CONF_REAL
#define Z_AXIS_DISCR       kine_param.Z_AXIS_DISCR   
{  
  int whale= (int) blockIdx.x;
  feasible =feasible + whale;
  
  int lastpnt,firstpnt;
  int current_conf       = 0;
  int cycth     = ceil(((float)MAX_RID_CONF_REAL/(float)blockDim.x));
  float tot     = 0;
  bool skip     = false;
  int neigh   = 0.0;
  
  Multi_Graph_Helper Graph(graph_ptr,gridDim.x,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL);
  Multi_Heur_Helper Heur(heur,gridDim.x,PATH_PNT,MAX_RID_CONF_REAL);
  
  if (feasible[0])//check feasibility TODO
  {
    for(int q=0;q<cycth;q++)
    {
      current_conf = threadIdx.x + q * blockDim.x;
      if(current_conf<MAX_RID_CONF_REAL)
      {
        if(Graph.getCh(whale,0,current_conf))
        {
          neigh = 1.0;
          neigh = Graph.getNeigh(whale,current_conf); 
          Heur.setHeuVal(whale,0,0,current_conf,(float)neigh);
        }
        else
        {
          Heur.setHeuVal(whale,0,0,current_conf,CUDART_NAN_F);
        }
        
        for(int pnt=1;pnt<PATH_PNT;pnt++)
        {
          for(int next_conf=0;next_conf<MAX_RID_CONF_REAL;next_conf++)
          {
            if( Graph.getCh(whale,pnt-1,current_conf) && Graph.getCh(whale,pnt,next_conf))
            {
              Heur.setHeuVal(whale,pnt,current_conf,next_conf,0.0);
              skip = false;
              for(int jnt=0;jnt<JOINTS_TOOL;jnt++)
              {
                if((fabs(Graph(whale,pnt-1,current_conf,jnt) - Graph(whale,pnt,next_conf,jnt))/timevector[pnt-1]) < robot2_speed_lim[jnt].upper)
                {
                  Heur.increaseHeu(whale,pnt,current_conf,next_conf, fabs(Graph(whale,pnt-1,current_conf,jnt) - Graph(whale,pnt,next_conf,jnt)));
                }
                else
                {
                  if((bool)multiturn_axis[jnt])
                  {
                    if(Graph(whale,pnt-1,current_conf,jnt)*Graph(whale,pnt,next_conf,jnt)<0 && 
                      (((2*PI_F- fabs(Graph(whale,pnt-1,current_conf,jnt) - Graph(whale,pnt,next_conf,jnt)))/timevector[pnt-1]) < robot2_speed_lim[jnt].upper))
                    {
                      Heur.increaseHeu(whale,pnt,current_conf,next_conf, 2*PI_F- fabs((Graph(whale,pnt-1,current_conf,jnt) - Graph(whale,pnt,next_conf,jnt))));
                    }
                    else
                    {
                      Heur.setHeuVal(whale,pnt,current_conf,next_conf,CUDART_NAN_F);
                      skip = true;
                      break;
                    }
                  }
                  else
                  {
                    Heur.setHeuVal(whale,pnt,current_conf,next_conf,CUDART_NAN_F);
                    skip = true;
                    break;
                  }
                }
              }
              
              if(skip) continue;
              
              if(Heur(whale,pnt,current_conf,next_conf)!=0 && Heur(whale,pnt,current_conf,next_conf)!=CUDART_NAN_F)
              {
                Heur.setHeuVal(whale,pnt,current_conf,next_conf,__powf(Heur(whale,pnt,current_conf,next_conf),(-beta)));
              }
            }
            else
            {
              Heur.setHeuVal(whale,pnt,current_conf,next_conf,CUDART_NAN_F);
            }
          }
        }
      }
    }
    
    for(int pnt=0;pnt<PATH_PNT;pnt++)
    {
      for(int q=0;q<cycth;q++)
      {
        current_conf = threadIdx.x + q * blockDim.x;
        if(current_conf<MAX_RID_CONF_REAL)
        {
          tot = 0;
          for(int next_conf=0;next_conf<MAX_RID_CONF_REAL;next_conf++) 
          {
            if(!isnan(Heur(whale,pnt,current_conf,next_conf))) tot+=  Heur(whale,pnt,current_conf,next_conf);
          }

          if(tot!=0)
          {
            for(int next_conf=0;next_conf<MAX_RID_CONF_REAL;next_conf++) 
            {
              Heur.setHeuVal(whale,pnt,current_conf,next_conf, MAX_RID_CONF_REAL*(Heur(whale,pnt,current_conf,next_conf)/tot));
            }
          }
        }
      }
    }
  }
#undef PATH_PNT
#undef JOINTS_TOOL
#undef JOINTS_POS
#undef MAX_RID_CONF
#undef MAX_RID_CONF_REAL
#undef Z_AXIS_DISCR
}
  
__global__ void AntCycle(void* graph,int* solution,int* sol,float* maxfobjtot,void* heur,bool* feasible,ACO_params ACO,
                         bool* coll_check,Kine_const kine_param, int* multiturn_axis, int* logantsdev,float* logphdev,
                         float* timevector, joint_boundary *joint_limits,float* real_joint_angle)
#define PATH_PNT           kine_param.PATH_PNT
#define JOINTS_POS         kine_param.JOINTS_POS
#define JOINTS_TOOL        kine_param.JOINTS_TOOL
#define MAX_RID_CONF       kine_param.MAX_RID_CONF
#define MAX_RID_CONF_REAL  kine_param.MAX_RID_CONF_REAL
{
  
  float rnd_sel = 0.0;
  float prev_ph = 0.0;
  curandState_t state;
  int actsol = 0; //TODO
  int whale  = blockIdx.x;
  int n_ants = blockDim.x;
  bool sol_found;
  bool first_set = true;

  
  extern __shared__ int shmemant[];
  float     *phobj         = (float  *)&shmemant[0];
  float     *phobjcpy      = (float  *)&phobj[n_ants];
  float     *tot_ph        = (float  *)&phobjcpy[n_ants];
  int       *ind           = (int    *)&tot_ph[n_ants];
  int       *indtot        = (int    *)&ind[1];
  float     *phmin         = (float  *)&indtot[1];
  float     *phmax         = (float  *)&phmin[1];
  int       *hold          = (int    *)&phmax[1];
  int       *holdtot       = (int    *)&hold[1];
  float     *maxfobjcycle  = (float  *)&holdtot[1];
  int       *best          = (int    *)&maxfobjcycle[1];
  int       *bestcyc       = (int    *)&best[1];
  float     *PHevap        = (float  *)&bestcyc[1];
  float     *nodes         = (float  *)&PHevap[1];
  
  maxfobjtot = maxfobjtot + whale;
  solution   = solution   + whale*PATH_PNT;
  sol        = sol        + whale*PATH_PNT*n_ants;
  feasible   = feasible   + whale;
  coll_check = coll_check + whale;
  
  
  Multi_Graph_Helper Graph(graph,gridDim.x,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL);
  Multi_Heur_Helper   Heur(heur,gridDim.x,PATH_PNT,MAX_RID_CONF_REAL);
  
  real_joint_angle = real_joint_angle + JOINTS_TOOL*blockDim.x*blockIdx.x + JOINTS_TOOL*threadIdx.x; 
 
  curand_init(clock64() ,threadIdx.x+blockDim.x*whale, clock64(), &state);

  int cycth     = (PATH_PNT/(int)blockDim.x) + 1;
  
  if(threadIdx.x==0)
  {
    ind[0]        = 0;
    indtot[0]     = 0;
    maxfobjtot[0] = -1.0;
    phmax[0]      = 1;
    phmin[0]      = 1;
    PHevap[0]     = 0;
    nodes[0]      = 0.0;
  }
  
  __syncthreads();
  
  for(int cy=0;cy<cycth;cy++)
  {
    int thcyc = threadIdx.x + cy * blockDim.x;
    if(thcyc<PATH_PNT)
    {
      for(int mm=0;mm<MAX_RID_CONF_REAL;mm++)
      {
        fAtomicAdd(nodes,(float)Graph.getCh(whale,thcyc,mm));
      }
    }
  }
  
  __syncthreads();
  
  if(threadIdx.x==0)
  {
    nodes[0] = nodes[0] / PATH_PNT;
    float num=1.0-pow(0.05,1.0/nodes[0]);
    float den=((nodes[0]*0.5)-1.0)*(pow(0.05,1.0/nodes[0]));
    PHevap[0] = 1.0 - pow((num/den),(1.0/(float)ACO.cycles));
  }

  __syncthreads();
  __threadfence();
  
  if (*feasible && (*coll_check)==false) //TODO formiche pd pd pd
  {
    for(int cyc=0;cyc<ACO.cycles;cyc++)
    {
      if(threadIdx.x==0)
      {
        maxfobjcycle[0] = -1.0;
        hold[0]         = 1;
        holdtot[0]      = 1;
        best[0]         = 0;
        bestcyc[0]      = 0;
      }
      actsol = 0;
      sol_found = true;
      __syncthreads(); 
      __threadfence();
      
      for (int pnt=0 ; pnt<PATH_PNT ; pnt++) //PROBABILISTIC SELECTION IMPLEMENTATION
      {
        
        sol[PATH_PNT * threadIdx.x + pnt] = -1;
        tot_ph[threadIdx.x] = 0.0;
        rnd_sel     = 0.0;
        
        __syncthreads();
        __threadfence();

        for(int next_conf=0;next_conf<MAX_RID_CONF_REAL;next_conf++)
        {
          tot_ph[threadIdx.x] += Heur(whale,pnt,actsol,next_conf) * Graph.getPh(whale,pnt,next_conf);
        }       
        
        __syncthreads();
        __threadfence();
        
        if(tot_ph[threadIdx.x] > 0)
        {
          rnd_sel = curand_uniform(&state) * tot_ph[threadIdx.x];
        }
        else
        {
          sol_found = false;
        }
        
        __syncthreads();
        __threadfence();
        
        prev_ph = 0.0;
        
        if(sol_found)
        {
          for(int conf=0;conf<MAX_RID_CONF_REAL;conf++)
          {
            prev_ph += Graph.getPh(whale,pnt,conf) * Heur(whale,pnt,actsol,conf);
            
            if(rnd_sel <= prev_ph  ) //&& prev_ph>0
            {
              sol[threadIdx.x*PATH_PNT + pnt] = conf;
              actsol = conf;
    #ifdef DEBUG_HEUR
              logantsdev[whale*n_ants*PATH_PNT*ACO.cycles + cyc*PATH_PNT*ACO.ants + threadIdx.x*PATH_PNT + pnt] = conf;
    #endif
              break;
            }

            if(conf==((MAX_RID_CONF_REAL)-1) && rnd_sel>prev_ph)
            {
              printf("ERROR in ACO %f>%f>%f thread : %d  block : %d cyc :%d conf : %d pnt :%d\n",rnd_sel,prev_ph,tot_ph[threadIdx.x],threadIdx.x,blockIdx.x,cyc,conf,pnt);
              actsol = conf;
              sol[threadIdx.x*PATH_PNT + pnt]=conf;
            }
          }
        }
      }
      
      phobj[threadIdx.x]     = 0.0;
      phobjcpy[threadIdx.x]  = 0.0;

      __syncthreads();
      __threadfence();
      
//       if(sol_found) EvalDist(sol,&phobj[threadIdx.x],Graph,kine_param, multiturn_axis);
      if(sol_found) Obj_fun_jointspeed_jointlimits(sol,&phobj[threadIdx.x],Graph,kine_param, multiturn_axis,timevector,joint_limits,real_joint_angle);
//       if(sol_found) printf("Obj func : %f    evap : %f    sol found : %d \n",phobj[threadIdx.x],ACO.PHevap,sol_found);
      phobjcpy[threadIdx.x] = phobj[threadIdx.x];
      
      __syncthreads();
      __threadfence();
      
      for (unsigned int s=blockDim.x/2; s>0; s>>=1)
      {
        if (threadIdx.x < s) phobjcpy[threadIdx.x] = max(phobjcpy[threadIdx.x], phobjcpy[threadIdx.x + s]);
        __syncthreads();
      }
      
      if (threadIdx.x == 0 && phobjcpy[0] > 0.0)
      {
        maxfobjcycle[0] = phobjcpy[0];
        bestcyc[0]      = 1;
        if(maxfobjtot[0] < maxfobjcycle[0])
        {
          best[0] = 1;
          maxfobjtot[0] = maxfobjcycle[0];
        }
        
        if(first_set)
        {
          phmax[0] = maxfobjtot[0]/PHevap[0];
          phmin[0] = maxfobjtot[0]/PHevap[0];
          first_set = false;
        }  
        else
        {
          phmax[0] = maxfobjtot[0]/PHevap[0];
          phmin[0] = (phmax[0]*(1.0-powf(0.05,1.0/nodes[0])))/(((nodes[0]/2.0)-1.0)*(powf(0.05,1.0/nodes[0])));
        }
//         printf("maxfobjcycle : %f    phobjcpy : %f  \n",maxfobjcycle[0],phobjcpy[0]);
      }
      
  #ifdef DEBUG_HEUR
  if(threadIdx.x==0)
  {
          logphdev[whale*PATH_PNT*ACO.cycles*MAX_RID_CONF + whale*3*ACO.cycles +
                    cyc*PATH_PNT*MAX_RID_CONF + cyc * 3]     = phmax[0];
          logphdev[whale*PATH_PNT*ACO.cycles*MAX_RID_CONF + whale*3*ACO.cycles +
                    cyc*PATH_PNT*MAX_RID_CONF + cyc * 3 + 1] = phmin[0];
          logphdev[whale*PATH_PNT*ACO.cycles*MAX_RID_CONF + whale*3*ACO.cycles +
                    cyc*PATH_PNT*MAX_RID_CONF + cyc * 3 + 2] = PHevap[0];
  }
  #endif
      
//       if(threadIdx.x==0) printf("whale : %d  cycle %d maxfobjcycle : %f  maxfobjtot : %f max-min PH : %f %f \n",whale,cyc,maxfobjcycle[0],maxfobjtot[0],phmax[0],phmin[0]);//TODO
      __syncthreads();
      
      //     BEST SOLUTION PICK
      if(maxfobjcycle[0]==phobj[threadIdx.x] && atomicExch(hold,(int)(maxfobjcycle[0]!=phobj[threadIdx.x])) && bestcyc[0])
      { 
        if(!sol_found) printf("NO SOL megapd1 whale : %d thread : %d cyc : %d maxfobjcycle : %f \n\n",whale,threadIdx.x,cyc,maxfobjcycle[0]);
        ind[0]=threadIdx.x;
      }
      
      if(maxfobjtot[0]==phobj[threadIdx.x] && atomicExch(holdtot,(int)(maxfobjtot[0]!=phobj[threadIdx.x])) && best[0])
      {
        if(!sol_found) printf("megapd2????????????????????????????????????????????????????????????????????????????\n\n");
        indtot[0] = threadIdx.x;
      }
      
      __syncthreads();
      __threadfence();
      
      //EVAPORATION
      for(int cy=0;cy<cycth;cy++)
      {
        int thcyc = threadIdx.x + cy * blockDim.x;
        if(thcyc<PATH_PNT)
        {
          for(int mm=0;mm<MAX_RID_CONF_REAL;mm++)
          {
            if(Graph.getCh(whale,thcyc,mm))
            {
              atomicMul(Graph.getPhAdd(whale,thcyc,mm),(1-PHevap[0]));
              if(Graph.getPh(whale,thcyc,mm) < phmin[0]) 
              {
                Graph.setPh(whale,thcyc,mm,phmin[0]);
              }
            }
          }
        }
      }
      
      __syncthreads();
      __threadfence();
      //PHERORMONE INCREASE
      for(int cy=0;cy<cycth;cy++)
      {
        int thcyc = threadIdx.x + cy * blockDim.x;
        if(thcyc<PATH_PNT)
        {
          fAtomicAdd(Graph.getPhAdd(whale,thcyc,sol[thcyc+PATH_PNT*ind[0]]),phobj[ind[0]]);
        }
      }
      
      __syncthreads();
      __threadfence();
      
      //PHERORMONE SATURATION
      for(int cy=0;cy<cycth;cy++)
      {
        int thcyc = threadIdx.x + cy * blockDim.x;
        if(thcyc<PATH_PNT)
        {
          for(int mm=0;mm<MAX_RID_CONF_REAL;mm++)
          {
            if(Graph.getCh(whale,thcyc,mm))
            {
              if(Graph.getPh(whale,thcyc,mm) < phmin[0]) 
              {
                Graph.setPh(whale,thcyc,mm,phmin[0]);
              }
              if(Graph.getPh(whale,thcyc,mm) > phmax[0]) 
              {
                Graph.setPh(whale,thcyc,mm,phmax[0]);
              }
            }
  #ifdef DEBUG_HEUR
            logphdev[whale*PATH_PNT*ACO.cycles*MAX_RID_CONF + whale*3*ACO.cycles +
                    cyc*PATH_PNT*MAX_RID_CONF + cyc * 3 +
                    thcyc*MAX_RID_CONF +
                    mm + 3] = Graph.getPh(whale,thcyc,mm);
  #endif
          }
        }
      }
    
      __syncthreads();
      __threadfence();
      
      //BEST SOLUTION 
      if(best[0]==1)
      {
        for(int cy=0;cy<cycth;cy++)
        {
          int thcyc = threadIdx.x + cy * blockDim.x;
          if(thcyc<PATH_PNT)
          {
            solution[thcyc] = sol[(indtot[0]) * PATH_PNT + thcyc]; 
          }
        }
        __syncthreads();
        __threadfence();
      }
    }
    
    if (threadIdx.x<PATH_PNT)//TODO if = -1 unfeasible??
    {
      if(maxfobjtot[0] < 0.0)
      {
        feasible[0]   = false;
        coll_check[0] = true;
        return;//TODO
      }
      
      if (Graph.getCh(whale,threadIdx.x,solution[threadIdx.x])==false)
      {
        printf("MEGA PDDDDDDD-- -- -- -- -- -- whale : %d \n\n",whale);
        maxfobjtot[0] = -1;
        feasible[0]   = false;
        coll_check[0] = true;
      }
    }
    __syncthreads();
    __threadfence();
  }
  
#undef PATH_PNT
#undef JOINTS_TOOL
#undef JOINTS_POS
#undef MAX_RID_CONF
#undef MAX_RID_CONF_REAL
}

__global__ void WOACycle(joint_boundary* robot_tool_jnt_lim, joint_boundary* robot_pos_jnt_lim ,int w_iter,Eigen::Matrix4f* pathrel,float* timevector
                        ,Eigen::Matrix4f RobotsTrans,Eigen::Matrix4f gripper_trans,void* graph,void* joints_pos
                        ,bool* checkreach, Kine_const kine_param, Eigen::Matrix4f* inv_tool_trans,
                        Eigen::Matrix4f* Robot_gripper,int* max_conf_graph_dev,double* solIK,float* hal1,float* hal2,float* hal3)
#define PATH_PNT              kine_param.PATH_PNT
#define JOINTS_TOOL           kine_param.JOINTS_TOOL
#define JOINTS_POS            kine_param.JOINTS_POS
#define MAX_RID_CONF          kine_param.MAX_RID_CONF
#define Z_AXIS_DISCR          kine_param.Z_AXIS_DISCR
#define STEP                  kine_param.STEP
#define LOWER_RANGE           kine_param.LOWER_RANGE
#define UPPER_RANGE           kine_param.UPPER_RANGE
#define MAX_OPEN_CONE_ANGLE   kine_param.MAX_OPEN_CONE_ANGLE
#define STEP_OPEN_CONE_ANGLE  kine_param.STEP_OPEN_CONE_ANGLE
#define N_OPEN_CONE           kine_param.N_OPEN_CONE
#define STEP_ROT_CONE         kine_param.STEP_ROT_CONE
#define N_ROT_CONE            kine_param.N_ROT_CONE
#define RID_CONF              kine_param.RID_CONF
{
  extern __shared__ int shmemant[];
  int  *pnt_countercheck   = (int  *)&shmemant[0];
  int  *currline           = (int  *)&pnt_countercheck[1];
  int  *max_tot            = (int  *)&currline[1];

  checkreach         = checkreach + blockIdx.x;
  Robot_gripper      = Robot_gripper + blockIdx.x;
  max_conf_graph_dev = max_conf_graph_dev + blockIdx.x; 
  solIK              = solIK             + blockIdx.x*blockDim.x*JOINTS_TOOL + threadIdx.x*JOINTS_TOOL;

  int whale=blockIdx.x;
  Eigen::Matrix4f  Pos_TCP;
  float rnd_joint_val;
  Eigen::Matrix4f path, T_Rot_z, T_Rot_x, T_Rot_axis;
  float cone_angle = 0.0;
  float r_x, r_y, r_z;
  float rot_cone_angle = 0.0;
  int nsolIK,rid_config,open_angle,rot_cone,rid_z;
  
  Eigen::Matrix4f ConfMat ;
  Multi_Positioner_Helper Joint_pos(joints_pos,gridDim.x,JOINTS_POS);
  Multi_Graph_Helper Graph(graph,gridDim.x,PATH_PNT,MAX_RID_CONF,JOINTS_TOOL);

  curandState_t state;
  positioner_robot::IkReal eerot[9],eetrans[3];
  tool_robot::IkReal       eerotik[9],eetransik[3];
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(clock64() ,id, 0, &state);
  bool bSuccess, robot2_reach;
  ikfast::IkSolutionListBase<tool_robot::IkReal> solutions;
  int poses_per_thread = ceil((float)RID_CONF/(float)blockDim.x);
  
  int index        = threadIdx.x*poses_per_thread;
  int pose         = index;
  int currline_loc = 0;
  
  positioner_robot::IkReal positioner_joints[10];
  __syncthreads();
  __threadfence();
  if(threadIdx.x==0)
  {

    *checkreach=true;

    if (w_iter==0)
    {
      for (int jnt=0; jnt<JOINTS_POS; jnt++)
      {
        rnd_joint_val = curand_uniform(&state)*(robot_pos_jnt_lim[jnt].upper-robot_pos_jnt_lim[jnt].lower)+robot_pos_jnt_lim[jnt].lower;
        Joint_pos.setJointVal(whale,jnt,rnd_joint_val);
      }
    }
    for (int jnt=0; jnt<JOINTS_POS; jnt++)
    {
      
      if ((bool)(Joint_pos(whale,jnt) < robot_pos_jnt_lim[jnt].lower) || (bool)(Joint_pos(whale,jnt) > robot_pos_jnt_lim[jnt].upper))
      {
        rnd_joint_val = curand_uniform(&state)*(robot_pos_jnt_lim[jnt].upper-robot_pos_jnt_lim[jnt].lower)+robot_pos_jnt_lim[jnt].lower;
        Joint_pos.setJointVal(whale,jnt,rnd_joint_val);
      }
      positioner_joints[jnt]=Joint_pos(whale,jnt);
    }
    positioner_robot::ComputeFk(positioner_joints,eetrans,eerot);
   
    Pos_TCP(0,0)=eerot[0];Pos_TCP(0,1)=eerot[1];Pos_TCP(0,2)=eerot[2];Pos_TCP(0,3)=eetrans[0];
    Pos_TCP(1,0)=eerot[3];Pos_TCP(1,1)=eerot[4];Pos_TCP(1,2)=eerot[5];Pos_TCP(1,3)=eetrans[1];
    Pos_TCP(2,0)=eerot[6];Pos_TCP(2,1)=eerot[7];Pos_TCP(2,2)=eerot[8];Pos_TCP(2,3)=eetrans[2];
    Pos_TCP(3,0)=0       ;Pos_TCP(3,1)=0       ;Pos_TCP(3,2)=0       ;Pos_TCP(3,3)=1;
    Robot_gripper[0]=RobotsTrans*Pos_TCP*gripper_trans;
    
    max_tot[0] = 0;
  }
  __threadfence();
  __syncthreads();
  for(int pnt=0; pnt<PATH_PNT; pnt++)
  {
    __syncthreads();
        
    if (threadIdx.x==0)
    {
      pnt_countercheck[0] = 0;
      currline[0]         = 0;
    }
    
    path = Robot_gripper[0]*pathrel[pnt];
    pose = index;
    
    __syncthreads();
    while (pose < index+poses_per_thread)
    {
//      printf("thread %d index %d iter %d pose %d posxthread %d ridconf %d blkdim %d\n",threadIdx.x,index,w_iter,pose,poses_per_thread,RID_CONF,blockDim.x);
      if (pose<RID_CONF)
      {
        open_angle = pose/(N_ROT_CONE*Z_AXIS_DISCR);
        if(N_OPEN_CONE > 1)
        {cone_angle = (MAX_OPEN_CONE_ANGLE/(N_OPEN_CONE-1))*hal2[open_angle];}
        else
        {cone_angle = 0;}

        
        T_Rot_x(0,0)= 1;  T_Rot_x(0,1)= 0;                  T_Rot_x(0,2)= 0;                  T_Rot_x(0,3)= 0;
        T_Rot_x(1,0)= 0;  T_Rot_x(1,1)= cosf(cone_angle);   T_Rot_x(1,2)=-sinf(cone_angle);   T_Rot_x(1,3)= 0; 
        T_Rot_x(2,0)= 0;  T_Rot_x(2,1)= sinf(cone_angle);   T_Rot_x(2,2)= cosf(cone_angle);   T_Rot_x(2,3)= 0;
        T_Rot_x(3,0)= 0;  T_Rot_x(3,1)= 0;                  T_Rot_x(3,2)= 0;                  T_Rot_x(3,3)= 1;
        
        r_x=T_Rot_x(2,0);
        r_y=T_Rot_x(2,1);
        r_z=T_Rot_x(2,2);
        
        rot_cone = (pose%(N_ROT_CONE*Z_AXIS_DISCR))/Z_AXIS_DISCR;
        rot_cone_angle=2*PI_F*hal3[rot_cone];
        
        T_Rot_axis(0,0)=r_x*r_x*(1-cosf(rot_cone_angle))+cosf(rot_cone_angle);  T_Rot_axis(0,1)=r_x*r_y*(1-cosf(rot_cone_angle))-r_z*sinf(rot_cone_angle);  
        T_Rot_axis(0,2)=r_x*r_z*(1-cosf(rot_cone_angle))+r_y*sinf(rot_cone_angle);  T_Rot_axis(0,3)=0;
        
        T_Rot_axis(1,0)=r_x*r_y*(1-cosf(rot_cone_angle))+r_z*sinf(rot_cone_angle);  T_Rot_axis(1,1)=r_y*r_y*(1-cosf(rot_cone_angle))+cosf(rot_cone_angle);
        T_Rot_axis(1,2)=r_y*r_z*(1-cosf(rot_cone_angle))-r_x*sinf(rot_cone_angle);  T_Rot_axis(1,3)=0;
        
        T_Rot_axis(2,0)=r_x*r_z*(1-cosf(rot_cone_angle))-r_y*sinf(rot_cone_angle);  T_Rot_axis(2,1)=r_y*r_z*(1-cosf(rot_cone_angle))+r_x*sinf(rot_cone_angle);
        T_Rot_axis(2,2)=r_z*r_z*(1-cosf(rot_cone_angle))+cosf(rot_cone_angle);  T_Rot_axis(2,3)=0;
        
        T_Rot_axis(3,0)=0;  T_Rot_axis(3,1)=0;  T_Rot_axis(3,2)=0;  T_Rot_axis(3,3)=1;
        
        rid_z= pose - open_angle*N_ROT_CONE*Z_AXIS_DISCR - rot_cone*Z_AXIS_DISCR;
//        printf("thread %d index %d iter %d pose %d  open angle %d cone angle %f rot cone %d rot cone angle %f rid_z %f \n",threadIdx.x,index,w_iter,pose,open_angle,cone_angle,rot_cone,rot_cone_angle,LOWER_RANGE+((UPPER_RANGE - LOWER_RANGE)*hal1[rid_z]));

        T_Rot_z(0,0)=cosf((LOWER_RANGE+(UPPER_RANGE - LOWER_RANGE)*hal1[rid_z])); T_Rot_z(0,1)=-sinf(LOWER_RANGE+((UPPER_RANGE - LOWER_RANGE)*hal1[rid_z]));T_Rot_z(0,2)=0;T_Rot_z(0,3)=0;
        T_Rot_z(1,0)=sinf((LOWER_RANGE+(UPPER_RANGE - LOWER_RANGE)*hal1[rid_z])); T_Rot_z(1,1)= cosf(LOWER_RANGE+((UPPER_RANGE - LOWER_RANGE)*hal1[rid_z]));T_Rot_z(1,2)=0;T_Rot_z(1,3)=0;
        T_Rot_z(2,0)=0;                            T_Rot_z(2,1)=0;                            T_Rot_z(2,2)=1;T_Rot_z(2,3)=0;
        T_Rot_z(3,0)=0;                            T_Rot_z(3,1)=0;                            T_Rot_z(3,2)=0;T_Rot_z(3,3)=1;
        
        ConfMat = path * T_Rot_x * T_Rot_axis * T_Rot_z * (inv_tool_trans[0]);//TODO shared o speed up?
        
        eerotik[0]=ConfMat(0,0);eerotik[1]=ConfMat(0,1);eerotik[2]=ConfMat(0,2);eetransik[0]=ConfMat(0,3);
        eerotik[3]=ConfMat(1,0);eerotik[4]=ConfMat(1,1);eerotik[5]=ConfMat(1,2);eetransik[1]=ConfMat(1,3);
        eerotik[6]=ConfMat(2,0);eerotik[7]=ConfMat(2,1);eerotik[8]=ConfMat(2,2);eetransik[2]=ConfMat(2,3);
        
        bSuccess = tool_robot::ComputeIk(eetransik,eerotik,NULL,solutions);
        if( bSuccess )
        {
          nsolIK = solutions.GetNumSolutions();
          for(std::size_t conf = 0; conf<nsolIK; conf++)
          {
            robot2_reach=true;
            const ikfast::IkSolutionBase<tool_robot::IkReal>& sol = solutions.GetSolution(conf); //UNISCI
            sol.GetSolution(&solIK[0],NULL);                          //UNISCI
            for (int jnt=0; jnt<JOINTS_TOOL; jnt++)
            {
              if ((solIK[jnt] < robot_tool_jnt_lim[jnt].lower) || (solIK[jnt] > robot_tool_jnt_lim[jnt].upper))
              {
                robot2_reach=false;
                break;
              }
            }
            
            if(robot2_reach==true)
            {
              atomicAdd(pnt_countercheck,1);
              currline_loc = atomicAdd(currline, 1);
              for (int jnt=0; jnt<JOINTS_TOOL; jnt++)
              {
                Graph.setJointVal(whale,pnt,currline_loc,jnt,solIK[jnt]);
              }
              Graph.setCh(whale,pnt,currline_loc,true);
              Graph.setPh(whale,pnt,currline_loc,1.0);
            }
          }
        }
      }
      __syncthreads();
      pose++;
    }
    __syncthreads();
    if(threadIdx.x==0)
    {
      atomicMax(max_tot,currline[0] - 1);
    }
    if(pnt_countercheck[0] == 0)
    {
      checkreach[0]=false;
      return;  
    }
  }
  __syncthreads();
  __threadfence();

  if(threadIdx.x == 0)
  {
    max_conf_graph_dev[0] = max_tot[0];
  }
  __syncthreads();
#undef RID_CONF
#undef PATH_PNT
#undef JOINTS_POS
#undef JOINTS_TOOL
#undef MAX_RID_CONF
#undef Z_AXIS_DISCR
#undef STEP
#undef LOWER_RANGE
#undef UPPER_RANGE
#undef MAX_OPEN_CONE_ANGLE 
#undef STEP_OPEN_CONE_ANGLE
#undef N_OPEN_CONE  
#undef STEP_ROT_CONE
#undef N_ROT_CONE   
}


__global__ void WOABestAndHeuristic(void *bestjointpos, void *joints_pos,int cyc, WOA_params WOA,float *WhaleScores, float *bestScore,int *Antpaths, 
                                    void *graph,void *best_path_angles, Kine_const kine_param, 
                                    joint_boundary* robot_tool_jnt_lim, int whale_check_coll)
{
  
  int best;
  int best_cycle;
  int whale=threadIdx.x;
  
  Multi_Positioner_Helper Joint_pos(joints_pos,whale,kine_param.JOINTS_POS);
  Multi_Positioner_Helper Best_Joint_pos(bestjointpos,1,kine_param.JOINTS_POS);
  Multi_Graph_Helper Best_Path_Angles(best_path_angles,1,kine_param.PATH_PNT,1,kine_param.JOINTS_TOOL);
  Multi_Graph_Helper Graph (graph,whale,kine_param.PATH_PNT,kine_param.MAX_RID_CONF,kine_param.JOINTS_TOOL);
   
//   if(threadIdx.x==0)
//   {
//     for(int whale=0;whale<blockDim.x;whale++)
//     {
//       for (int pnt=0; pnt<kine_param.PATH_PNT; pnt++)
//       {
//         printf("%+i\t",Antpaths[whale*kine_param.PATH_PNT+pnt]);
//       }
//       printf("%+f\n",WhaleScores[whale]);
//     }
//   }
  
  if (threadIdx.x==0)
  {
    if ( whale_check_coll>=0 &&  WhaleScores[whale_check_coll]>0)
    {
      *bestScore=WhaleScores[whale_check_coll];
      for (int jnt=0; jnt<kine_param.JOINTS_POS; jnt++)
      {
        Best_Joint_pos.setJointVal(0,jnt,Joint_pos(whale_check_coll,jnt));
      }
      for (int pnt=0; pnt<kine_param.PATH_PNT; pnt++)
      {
        for (int jnt=0; jnt<kine_param.JOINTS_TOOL; jnt++)
        {
          Best_Path_Angles.setJointVal(0,pnt,0,jnt,Graph(whale_check_coll,pnt,Antpaths[whale_check_coll*kine_param.PATH_PNT + pnt],jnt));
        }
      }
    }
  }
  
  __syncthreads();
  __threadfence();

  float a,a2,A,C,b,l,D_X_rand,X_rand;
  
  curandState_t state;
  curand_init(clock64() ,threadIdx.x, 0, &state);
  
  a = 2-(float)cyc*((2.0) / (float)WOA.cycles);
  a2=-1+(float)cyc*((-1.0)/(float)WOA.cycles);
  A=2*a*curand_uniform(&state)-a;
  C=2*curand_uniform(&state);

  b=1;
  l=(a2-1)*curand_uniform(&state)+1;
  
  float p=curand_uniform(&state);
  float rnd_joint_val;
  
  
  for(int jnt=0;jnt<kine_param.JOINTS_POS;jnt++)
  {
    if (*bestScore==-1.0)
    {
      rnd_joint_val = curand_uniform(&state)*(robot_tool_jnt_lim[jnt].upper-robot_tool_jnt_lim[jnt].lower)+robot_tool_jnt_lim[jnt].lower;
      Joint_pos.setJointVal(whale,jnt,rnd_joint_val);
    }
    else
    {
      if(p<0.5)
      {
        if(fabsf(A)>=WOA.A_lim)
        {
          X_rand = Joint_pos(floor((blockDim.x)*curand_uniform(&state)),jnt) ; //blockDim.y e z??
          D_X_rand=abs(C*X_rand-Joint_pos(whale,jnt));
          Joint_pos.setJointVal(whale,jnt,X_rand-A*D_X_rand);
        }
        else
        {
          Joint_pos.setJointVal(whale,jnt, Best_Joint_pos(0,jnt)-A*abs(C*Best_Joint_pos(0,jnt)-Joint_pos(whale,jnt)));
        }
      }
      else
      {
        Joint_pos.setJointVal(whale,jnt, abs(Best_Joint_pos(0,jnt)-Joint_pos(whale,jnt)*exp(b*l)*cos(l*2*PI_F)+Best_Joint_pos(0,jnt)));
      }
    }
  }
  __syncthreads();
  __threadfence();
}
