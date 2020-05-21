#pragma once
#ifndef ECUDA_VECTOR_HPP
#define ECUDA_VECTOR_HPP

#include <vector>
#include <assert.h>

namespace cuda_std {

#define MAX_VECTOR_DIM 8 ///CORREGGERE ASSOLUTAMENTE
  template< typename T  >
  class vector
  {
  private:
    T 			val_[MAX_VECTOR_DIM];
    std::size_t 	dim_;
  public:
    __host__ __device__ explicit vector()
    {
      clear();
    }
    __host__ __device__ vector(std::size_t dim)
    {
      bool ok_vector = ( dim <= MAX_VECTOR_DIM );
      assert( ok_vector );
      resize(dim);
    }
    __host__ __device__ vector(const cuda_std::vector<T>& v)
    {
      dim_ = v.size();
      for( size_t i=0; i< dim_; i++)
	val_[i] = v[i];
    }
    __host__ __device__ void clear()
    {
//       memset( val_, 0x0, MAX_VECTOR_DIM * sizeof( T ) );
      dim_ = 0;
    }
    __host__ __device__ void resize(std::size_t dim)
    {
      bool ok_resize = ( dim <= MAX_VECTOR_DIM );
      assert( ok_resize );
      dim_ = dim;
//       memset( val_, 0x0, MAX_VECTOR_DIM * sizeof( T ) );
    }
    __host__ __device__ T& at(std::size_t i)
    {
      bool ok_at = ( i < dim_ );
      assert( ok_at );
      return val_[i];
    }
    __host__ __device__ const T& at(std::size_t i) const
    {
      bool ok_at_const = ( i < dim_ );
      assert( ok_at_const );
      return val_[i];
    }
    __host__ __device__ T& operator[](std::size_t i) 
    {
      bool ok_operator = ( i < dim_ );
      assert( ok_operator );
      return val_[i];
    }
    __host__ __device__ const T& operator[](std::size_t i) const
    {
      bool ok_operator_const = (  i < dim_ );
      assert( ok_operator_const );
      return val_[i];
    }
    __host__ __device__ void push_back(const T& element )
    {
      dim_++;
      val_[dim_-1] = element;
    }
    __host__ __device__ std::size_t size() const 
    {
      return dim_;
    }
    
    __host__ __device__ T* data() 
    {
     return &val_[0]; 
    }

    __host__ __device__ const T* data() const 
    {
      return &val_[0];
    }
    __host__ __device__ cuda_std::vector<T>& operator=(const cuda_std::vector<T>& v)
    {
      dim_ = v.size();
      for( size_t i=0; i< dim_; i++)
	val_[i] = v[i];
      return *this;
    }
    
    
}; // namespace ecuda
}
#endif