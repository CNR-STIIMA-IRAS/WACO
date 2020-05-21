#ifndef SAVE_DATA_H
#define SAVE_DATA_H
#include "stdio.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <Eigen/Dense>



class save_data
{
  std::string folder;
  
public:
  save_data(const std::string folder_ext);
  void print_csv_JointPositioner(const std::string file, const int joints, void* values);
  void print_csv_JointRobot(const std::string file, const int path_points, const int joints, void* Rob);
  void print_csv_Float(const std::string file, float value);
  void print_Matrix_4x4(const std::string file, const int n_matrix ,Eigen::Matrix4f* matrix_array);
  void print_csv_fobj_history(const std::string file, float* fobj_history, int whale_iter, int n_whales);

  
};


#endif // SAVE_DATA_H
