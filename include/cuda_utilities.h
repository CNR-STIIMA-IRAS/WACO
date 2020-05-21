#ifndef __CUDA_UTILS_
#define __CUDA_UTILS_
#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static void __device__ HandleErrorDev( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERRORDEV( err ) (HandleErrorDev( err, __FILE__, __LINE__ ))


static void __host__ HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s %d in %s at line %d\n", cudaGetErrorString( err ), err,
                file, line );
        return;
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ __forceinline__ float atomicMul(float* address, float val)
{
  int32_t* address_as_int = reinterpret_cast<int32_t*>(address);
  int32_t old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ __forceinline__ static float fatomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,__float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ static float fatomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,__float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


__device__ __forceinline__ void fAtomicAdd (float *address, float value)
{
  int oldval, newval, readback;

  oldval = __float_as_int(*address);
  newval = __float_as_int(__int_as_float(oldval) + value);
  while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) 
    {
    oldval = readback;
    newval = __float_as_int(__int_as_float(oldval) + value);
    }
}

struct Multi_Graph_Helper
{
  void* cudaMemPointer_;
  const int ndof_;
  const int nconf_;
  const int npts_;
  const int ngraph_;
  const size_t sizej_;
  
  __host__ __device__ Multi_Graph_Helper( void* cudaMemPointer, const int ngraph, const int npts, const int nconf, const int ndof )
    : cudaMemPointer_(cudaMemPointer), ndof_(ndof), nconf_(nconf), npts_(npts), ngraph_(ngraph)
    ,sizej_((sizeof(float) * ndof) + (4 * sizeof(bool)) + sizeof(float) + sizeof(float)) {}
    
  __host__ __device__ void setJointVal( const int graph, const int iPnt, const int iConf, const int iJoint, const float val )
  {
    (*((float*)(static_cast<char*>(cudaMemPointer_) + graph*getSizeGraph() + iPnt*nconf_*sizej_ + iConf*sizej_ + iJoint * sizeof(float) ))) = val;
  }
  
  __host__ __device__ void setUnfeas(const int graph, const int iPnt, const int iConf)
  {
    for (int j=0; j<ndof_; j++)
    {
      setJointVal(graph,iPnt,iConf,j,NAN);
    }
    setCh(graph,iPnt,iConf,false);
    setPh(graph,iPnt,iConf,0.0);
  }
  
  __host__ __device__ void setCh( const int graph, const int iPnt, const int iConf, const bool ch )
  {
    (*((bool*)(static_cast<char*>(cudaMemPointer_) + graph*npts_ * nconf_ * sizej_ + iPnt*nconf_*sizej_ + iConf*sizej_ + ndof_ * sizeof(float) ))) = ch;
  }
  
   __host__ __device__ void setPh( const int graph, const int iPnt, const int iConf, const float ph )
  {
    (*((float*)(static_cast<char*>(cudaMemPointer_) + graph*npts_ * nconf_ * sizej_ + iPnt*nconf_*sizej_ + iConf*sizej_ + ndof_ * sizeof(float) + 4*sizeof(bool) ))) = ph;
  }
  
  __host__ __device__ void setTime( const int graph, const int iPnt, const int iConf, const float time )
  {
    (*((float*)(static_cast<char*>(cudaMemPointer_) + graph*getSizeGraph() + iPnt*nconf_*sizej_ + iConf*sizej_ + (ndof_ + 1) * sizeof(float) + 4*sizeof(bool) ))) = time;
  }
  
  __host__ __device__ float operator()(const int graph, const int iPnt, const int iConf, const int iJoint) const 
  { 
    return (*((float*)(static_cast<char*>(cudaMemPointer_) + graph*npts_*nconf_*sizej_ + iPnt*nconf_*sizej_ + iConf*sizej_ + iJoint * sizeof(float) )));
  }

  __host__ __device__ bool  getCh(const int graph, const int iPnt, const int iConf)
  {
    return (*((bool* )(static_cast<char*>(cudaMemPointer_) + graph*npts_ * nconf_ * sizej_ + iPnt*nconf_*sizej_ + iConf*sizej_ + ndof_  * sizeof(float) )));
  }

  __host__ __device__ float getPh(const int graph, const int iPnt, const int iConf)
  {
    return (*((float*)(static_cast<char*>(cudaMemPointer_) + graph*npts_ * nconf_ * sizej_ + iPnt*nconf_*sizej_ + iConf*sizej_ + ndof_  * sizeof(float) + 4*sizeof(bool) )));
  }
  
  __host__ __device__ float getTime(const int graph, const int iPnt, const int iConf)
  {
    return (*((float*)(static_cast<char*>(cudaMemPointer_) + graph*getSizeGraph() + iPnt*nconf_*sizej_ + iConf*sizej_ + (ndof_ + 1)  * sizeof(float) + 4*sizeof(bool) )));
  }
  
  __host__ __device__ float* getPhAdd(const int graph, const int iPnt, const int iConf)
  {
    return    (float*)((static_cast<char*>(cudaMemPointer_) + graph*npts_ * nconf_ * sizej_ + iPnt*nconf_*sizej_ + iConf*sizej_ + (ndof_ + 1)  * sizeof(float) ) );
  }
  
  __host__ __device__ float* getJointsAddr( const int graph,const int iPnt, const int iConf)
  {
    return    (float*)((static_cast<char*>(cudaMemPointer_) + graph*npts_ * nconf_ * sizej_ + iPnt*nconf_*sizej_ + iConf*sizej_ ) );
  }
  
  __host__ __device__ size_t getSizeGraph()
  {
     return npts_ * nconf_ * sizej_;
  }
  
  __host__ __device__ size_t getSizeMultiGraph()
  {
    return ngraph_ * npts_ * nconf_ * sizej_;
  }
  
  __host__ __device__ int getNeigh( const int graph, const int iConf)//TODO better
  {
    int neigh = 0.0;
    for(int pnt=0;pnt<npts_;pnt++)
    {
      neigh += (*((int* )(static_cast<char*>(cudaMemPointer_) + graph*npts_ * nconf_ * sizej_ + pnt*nconf_*sizej_ + iConf*sizej_ + ndof_  * sizeof(float) )));
    }

    return neigh;
  }
  
  __host__ void dump_log(std::string path,int n_whales_,int iter)
  {
    std::string::iterator it = path.end() - 1;
    if (*it != '/') path = path + "/" ;
    std::ofstream log_file_joints;
    std::string path_joints = path + "joints.txt" ;
    log_file_joints.open(path_joints.c_str(), std::ios_base::app );
    
    log_file_joints << "Whale cycle : " << iter << std::endl << std::endl;
    for (int i_whale=0; i_whale<n_whales_; i_whale++)
    {
      log_file_joints << "Whale number : " << i_whale << std::endl << std::endl;
      for (int pnt=0; pnt<npts_; pnt++)
      {
        for(int jnt=0;jnt<ndof_;jnt++)
        {
          for (int conf=0; conf< nconf_; conf++)
          {
            log_file_joints << std::setiosflags(std::ios::fixed)
                            << std::setprecision(2)
                            << std::setw(5)
                            << std::left
                            << (*((float*)(static_cast<char*>(cudaMemPointer_) + i_whale*npts_*nconf_*sizej_ + pnt*nconf_*sizej_ + conf*sizej_ + jnt * sizeof(float) )))
                            << "   " ;
          }
          log_file_joints << std::endl;
        }
        log_file_joints << std::endl << std::endl;
      }
      log_file_joints << std::endl << std::endl << std::endl << std::endl;
    }
    
    std::ofstream log_file_check;
    std::string path_check = path + "check.txt" ;
    log_file_check.open(path_check.c_str(), std::ios_base::app );
    
    log_file_check << "Whale cycle : " << iter << std::endl << std::endl;
    for (int i_whale=0; i_whale<n_whales_; i_whale++)
    {
      log_file_check << "Whale number : " << i_whale << std::endl << std::endl;
      for (int pnt=0; pnt<npts_; pnt++)
      {
        for (int conf=0; conf< nconf_; conf++)
        {
          log_file_check << std::setiosflags(std::ios::fixed)
                         << std::setprecision(1)
                         << std::setw(2)
                         << std::left
                         << this->getCh(i_whale,pnt,conf)
                         << "  " ;
        }
        log_file_check << std::endl;
      }
      log_file_check << std::endl << std::endl << std::endl << std::endl;
    }
    
  }
};

struct Multi_Positioner_Helper
{
  void* cudaMemPointer_;
  const int ndof_;
  const size_t size_;
  const int ngraph_;
  
  __host__ __device__ Multi_Positioner_Helper( void* cudaMemPointer,const int ngraph, const int ndof )
    : cudaMemPointer_(cudaMemPointer), ndof_(ndof) ,size_((sizeof(float) * ndof)), ngraph_(ngraph) {}
    
    
  __host__ __device__ void setJointVal(const int graph, const int iJoint, const float val )
  {
    (*((float*)(static_cast<char*>(cudaMemPointer_) + graph*size_ + iJoint * sizeof(float) ))) = val;
  }
  
  __host__ __device__ float operator()(const int graph,const int iJoint) const 
  { 
    return (*((float*)(static_cast<char*>(cudaMemPointer_) + graph*size_ + iJoint * sizeof(float) )));
  }
  
  __host__ __device__ size_t getsizeMulti()
  {
    return size_ * ngraph_;
  }
  
  __host__ void dump_log(std::string path,int n_whales_,int iter)
  {
    std::string::iterator it = path.end() - 1;
    if (*it != '/') path = path + "/" ;
    std::ofstream log_pos_joints;
    std::string path_joints = path + "positioner.txt" ;
    log_pos_joints.open(path_joints.c_str(), std::ios_base::app );
    
//     log_pos_joints << "Whale cycle : " << iter << std::endl << std::endl;
    for (int i_whale=0; i_whale<n_whales_; i_whale++)
    {
      for(int jnt=0;jnt<ndof_;jnt++)
      {
        log_pos_joints << std::setiosflags(std::ios::fixed)
                        << std::setprecision(2)
                        << std::setw(5)
                        << std::left
                        << (*((float*)(static_cast<char*>(cudaMemPointer_) + i_whale*size_ +  jnt * sizeof(float) )))
                        << "   " ;
      }
      log_pos_joints << std::endl;
    }
//     log_pos_joints << std::endl << std::endl;
  }
};

struct Multi_Heur_Helper
{
  void* cudaMemPointer;
  const int max_rid_conf;
  const int ngraph;
  const int path_points;
  const size_t size_heu;
  
  __host__ __device__ Multi_Heur_Helper(void* cudaMemPointer_,const int ngraph_, const int path_points_, const int max_rid_conf_)
  : cudaMemPointer(cudaMemPointer_), max_rid_conf(max_rid_conf_), ngraph(ngraph_), path_points(path_points_),
  size_heu(sizeof(float)*path_points*max_rid_conf*max_rid_conf) {}
  
  __host__ __device__ float operator()(const int graph, const int point, const int current_conf, const int next_conf) const
  {
    if(!isnan((*((float*)(static_cast<char*>(cudaMemPointer) + ((graph*path_points*max_rid_conf*max_rid_conf) + (point*max_rid_conf*max_rid_conf) + (current_conf*max_rid_conf) + next_conf)*sizeof(float))))))
    {
      return (*((float*)(static_cast<char*>(cudaMemPointer) + ((graph*path_points*max_rid_conf*max_rid_conf) + (point*max_rid_conf*max_rid_conf) + (current_conf*max_rid_conf) + next_conf)*sizeof(float))));
    }
    else
    {
      return 0.0;
    }
      
    }
  
  __host__ __device__ void setHeuVal(const int graph, const int point, const int current_conf, const int next_conf, const float value)
  {
    (*((float*)(static_cast<char*>(cudaMemPointer) + ((graph*path_points*max_rid_conf*max_rid_conf) +(point*max_rid_conf*max_rid_conf) + (current_conf*max_rid_conf) + next_conf)*sizeof(float))))=value;
  }
  
  __host__ __device__ void increaseHeu(const int graph, const int point, const int current_conf, const int next_conf, const float value)
  {
    (*((float*)(static_cast<char*>(cudaMemPointer) + ((graph*path_points*max_rid_conf*max_rid_conf) + (point*max_rid_conf*max_rid_conf) + (current_conf*max_rid_conf) + next_conf)*sizeof(float)))) += value; 
  }
  
  __host__ __device__ size_t getsizeMulti()
  {
    return ngraph*size_heu;
  }
  
  __host__ void dump_log(std::string path,int n_whales_,int iter)
  {
    std::string::iterator it = path.end() - 1;
    if (*it != '/') path = path + "/" ;
    std::ofstream log_file_heur;
    std::string path_joints = path + "heur_" + std::to_string(iter) + ".txt" ;
    log_file_heur.open(path_joints.c_str(), std::ios_base::app );
    
    log_file_heur << "Whale cycle : " << iter << std::endl << std::endl;
    for (int i_whale=0; i_whale<n_whales_; i_whale++)
    {
      log_file_heur << "Whale number : " << i_whale << std::endl << std::endl;
      for (int pnt=0; pnt<path_points; pnt++)
      {
        log_file_heur << "Point : " << pnt << std::endl << std::endl;
        for(int conf0=0;conf0<max_rid_conf;conf0++)
        {
          for (int conf=0; conf< max_rid_conf; conf++)
          {
            log_file_heur << std::setiosflags(std::ios::fixed)
                          << std::setprecision(2)
                          << std::setw(5)
                          << std::left
                          << (*((float*)(static_cast<char*>(cudaMemPointer) + ((i_whale*path_points*max_rid_conf*max_rid_conf) + (pnt*max_rid_conf*max_rid_conf) + (conf0*max_rid_conf) + conf)*sizeof(float))))
                          << "   " ;
          }
          log_file_heur << std::endl;
        }
        log_file_heur << std::endl << std::endl;
      }
      log_file_heur << std::endl << std::endl << std::endl << std::endl;
    }
  }
};

#endif
