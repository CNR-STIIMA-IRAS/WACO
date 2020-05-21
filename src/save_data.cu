#include "save_data.h"
#include <roswacocuda.h>
#include <boost/filesystem.hpp>

save_data::save_data(const std::string folder_ext)
: folder(folder_ext)
{
  boost::filesystem::create_directories(folder.c_str());
//   char folder_char[1024];
//   strcpy(folder_char,folder.c_str());
//   mkdir(folder_char,S_IRWXU);
};

void save_data::print_csv_JointPositioner(const std::string file, const int joints, void* Pos)
{
  std::ofstream print_file;
  char file_path[1024];
  std::string file_path_str=folder+file;
  strcpy(file_path,file_path_str.c_str());
  print_file.open(file_path, std::ofstream::out | std::ofstream::trunc);
  
  Multi_Positioner_Helper Positioner(Pos,1,joints);
  
  for(int jnt=0; jnt<joints; jnt++)
  {
    print_file << Positioner(0,jnt) << "\t" ;
    
  }
  print_file.close();
}

void save_data::print_csv_JointRobot(const std::string file, const int path_points, const int joints, void* Rob)
{
  std::ofstream print_file;
  char file_path[1024];
  std::string file_path_str=folder+file;
  strcpy(file_path,file_path_str.c_str());
  print_file.open(file_path, std::ofstream::out | std::ofstream::trunc);
  
  Multi_Graph_Helper Robot(Rob,1,path_points,1,joints);
  
  for(int pnt=0;pnt<path_points; pnt++)
  {
    for(int jnt=0; jnt<joints; jnt++)
    {
      print_file << Robot(0,pnt,0,jnt) << "\t";
    }
    print_file << "\n";
  }
  
}

void save_data::print_csv_Float(const std::string file, float value)
{
  std::ofstream print_file;
  char file_path[1024];
  std::string file_path_str=folder+file;
  strcpy(file_path,file_path_str.c_str());
  print_file.open(file_path, std::ofstream::out | std::ofstream::trunc);
  
  print_file << value << "\n";
  
  print_file.close();
}

void save_data::print_Matrix_4x4(const std::string file, const int n_matrix ,Eigen::Matrix4f* matrix_array)
{
  std::ofstream print_file;
  char file_path[1024];
  std::string file_path_str=folder+file;
  strcpy(file_path,file_path_str.c_str());
  print_file.open(file_path, std::ofstream::out | std::ofstream::trunc);
  
  for (int i=0;i<n_matrix; i++)
  {
    for(int line=0; line<4; line++)
    {
      for (int row=0; row<4; row++)
      {
        print_file << matrix_array[i](line,row);
        if (row==3) print_file << ";";
        else print_file << ",";
      }
      print_file << "\n";
    }
    if (i==(n_matrix-1)) print_file << "%%%%%%%%%%%%%%\n";
    else print_file << "-------------\n";
  }
}

void save_data::print_csv_fobj_history(const std::string file, float* fobj_history, int whale_iter, int n_whales)
 {
    std::ofstream print_file;
    char file_path[1024];
    std::string file_path_str=folder+file;
    strcpy(file_path,file_path_str.c_str());
    print_file.open(file_path, std::ofstream::out | std::ofstream::trunc);
    
    for (int i=0; i<whale_iter; i++)
    {
      for (int w=0;w<(n_whales-1);w++)
      {
        print_file << fobj_history[i*n_whales + w] << ",";
      }
      print_file << fobj_history[i*n_whales + n_whales-1]<< "\n";
    }
    print_file.close();
 }