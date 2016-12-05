/**
*三维切分
**/
/****
*对数据进行三维划分
**/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#define TIME(a,b) (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))

extern int Init(double *data, long long L);
extern int Check(double *data, long long L);

typedef struct
{ 
  int nx, ny, nz;
} Info;

Info setup(int NX, int NY, int NZ, int P)
{ 
  /*Info result; 
  int myrank = 0;
  result.nx = NX;
  result.ny = NY;
  result.nz = NZ;
   */
  Info result; 
  int myrank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if(myrank == 0)
  { 
    result.nx = NX/2+NX%2;
    result.ny = NY/2+NY%2;
    result.nz = NZ/2+NZ%2;
  }
  else
  { 
    result.nx = NX/2;
    result.ny = NY/2;
    result.nz = NZ/2;
  }
  return result;
}

// 中心计算
int centerCal(double *rankA,double *rankB,int nx,int ny,int nz,int myrank)
{   
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))
  int flag1,flag2,flag3;
  flag1=(myrank-1)/4; flag2=((myrank-1)%4)/2; flag3=(myrank-1)%2;
  int s_x,e_x,s_y,e_y,s_z,e_z;
  s_x=0;  e_x=nx-1;
  s_y=0;  e_y=ny-1;
  s_z=0;  e_z=nz-1;
  if(flag1==1){
    s_x=1;  e_x=nx;
  }
  if(flag2==1){
    s_y=1;  e_y=ny;
  }
  if(flag3==1){
    s_z=1;  e_z=nz;
  }

    int i, j, k;
    #pragma omp parallel for schedule(dynamic)
    for (i = s_x; i < e_x; i++)
    { 
      for(j = s_y; j < e_y; j ++) { 
        for(k = s_z; k < e_z; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          rankB[RDX(i,j,k)] = r;
       }
      }
   }
    return 0;
}

//bound Task 24个面的更新
int boundCalTask1(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz)
{
  int i,j,k;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))
  //  printf("----Task1\n");   
    //the first: data from 5(yzBuf)
    i=nx-1;
    #pragma omp parallel for 
    for(j = 0; j < ny; j ++) { 
      for(k = 0; k < nz; k ++) {
            double r = 0.4*rankA[RDX(i,j,k)];
      if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else{
          r+=0.1*xyBuf[(nx-1)*ny+j];
      }          
      if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
      else{
          r+=0.1*xzBuf[(nx-1)*nz+k];
      }         
      r += 0.1*rankA[RDX(i-1,j,k)];
      r += 0.1*yzBuf[j*nz+k];
      rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 2(xyBuf)
    k=nz-1;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(j = 0; j < ny; j ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else{
            r+=0.1*xzBuf[(i+1)*nz-1];
          }          
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else{
            r+=0.1*yzBuf[(j+1)*nz-1];
          } 
          r += 0.1*rankA[RDX(i,j,k-1)];
          r+=0.1*xyBuf[i*ny+j];
          rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 3(xzbuf)
    j=ny-1;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(k = 0; k < nz; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else{
          r+=0.1*xyBuf[(i+1)*ny-1];
      }

          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else{
            r+=0.1*yzBuf[(ny-1)*nz+k];
          } 
          r += 0.1*rankA[RDX(i,j-1,k)];
          r+=0.1*xzBuf[i*nz+k];
          rankB[RDX(i,j,k)] = r;
    }
  }
   return 0;
}

int boundCalTask2(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz)
{
  int i,j,k;
  //int ind;
  int xyInd,xzInd,yzInd;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))
    //the first: data from 6(yzBuf)
   // ind=0;
    i=nx-1;
    #pragma omp parallel for
    for(j = 0; j < ny; j ++) { 
      for(k = 0; k < nz; k ++) {
            double r = 0.4*rankA[RDX(i,j,k)];
      if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else{
         r+=0.1*xyBuf[(nx-1)*ny+j];
      }   
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else r += 0.1*rankA[RDX(i,j,k)];        
      if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
      else{
          r+=0.1*xzBuf[(nx-1)*nz+k];
      }         
      r += 0.1*rankA[RDX(i-1,j,k)];
      r += 0.1*yzBuf[j*nz+k];
     // ind++;
      rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 1(xyBuf)
   // ind=0;
    k=0;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(j = 0; j < ny; j ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else   r += 0.1*rankA[RDX(i,j,k)];
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else{
            r+=0.1*xzBuf[i*nz];
            
          }          
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else{
            r+=0.1*yzBuf[j*nz];
          } 
          r += 0.1*rankA[RDX(i,j,k+1)];
          r+=0.1*xyBuf[i*ny+j];
   //       ind++;
          rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 3(xzbuf)
    // ind=0;
    j=ny-1;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(k = 0; k < nz; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else{
         r+=0.1*xyBuf[(i+1)*ny-1];
      }         
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else r += 0.1*rankA[RDX(i,j,k)];

          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else{
            r+=0.1*yzBuf[(ny-1)*nz+k];
          } 
          r += 0.1*rankA[RDX(i,j-1,k)];
          r+=0.1*xzBuf[i*nz+k];
  //        ind++;
          rankB[RDX(i,j,k)] = r;
    }
  }
   return 0;
}
int boundCalTask3(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz)
{
  int i,j,k;
  //int ind;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))

    //the first: data from 7(yzBuf)
    //ind=0;
    i=nx-1;
    #pragma omp parallel for
    for(j = 0; j < ny; j ++) { 
      for(k = 0; k < nz; k ++) {
            double r = 0.4*rankA[RDX(i,j,k)];
      if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else{
          r+=0.1*xyBuf[(nx-1)*ny+j];
      }          
      if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
      else{
        r+=0.1*xzBuf[(nx-1)*nz+k];
      }   
      if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
      else  r += 0.1*rankA[RDX(i,j,k)];
      r += 0.1*rankA[RDX(i-1,j,k)];
      r += 0.1*yzBuf[j*nz+k];
   //   ind++;
      rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 4(xyBuf)
    //ind=0;
    k=nz-1;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(j = 0; j < ny; j ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else{
            r+=0.1*xzBuf[(i+1)*nz-1];
          } 
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else  r+=0.1*rankA[RDX(i,j,k)];        
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else{
            r+=0.1*yzBuf[(j+1)*nz-1];
          } 
          r += 0.1*rankA[RDX(i,j,k-1)];
          r+=0.1*xyBuf[i*ny+j];
     //     ind++;

          rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 1(xzbuf)
  // ind=0;
    j=0;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(k = 0; k < nz; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else{
          r+=0.1*xyBuf[i*ny];
      }

          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else{
            r+=0.1*yzBuf[k];
          } 
          r += 0.1*rankA[RDX(i,j+1,k)];
          r+=0.1*xzBuf[i*nz+k];
   //       ind++;
          rankB[RDX(i,j,k)] = r;
    }
  }
   return 0;
}
int boundCalTask4(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz)
{
  int i,j,k;
 // int ind;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))

    //the first: data from 5(yzBuf)
   // ind=0;
    i=nx-1;
    #pragma omp parallel for
    for(j = 0; j < ny; j ++) { 
      for(k = 0; k < nz; k ++) {
            double r = 0.4*rankA[RDX(i,j,k)];
      if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else{
         r+=0.1*xyBuf[(nx-1)*ny+j];
      }   
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else  r+=0.1*rankA[RDX(i,j,k)];       
      if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
      else{
        r+=0.1*xzBuf[(nx-1)*nz+k];
      }         
      if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
      else r += 0.1*rankA[RDX(i,j,k)];
      r += 0.1*rankA[RDX(i-1,j,k)];
      r += 0.1*yzBuf[j*nz+k];
 //     ind++;
      rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 3(xyBuf)
   // ind=0;
    k=0;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(j = 0; j < ny; j ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else{
            r+=0.1*xzBuf[i*nz];
          }         
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else   r += 0.1*rankA[RDX(i,j,k)];      
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else{
            r+=0.1*yzBuf[j*nz];
          } 
          r += 0.1*rankA[RDX(i,j,k+1)];
          r+=0.1*xyBuf[i*ny+j];
   //       ind++;

         rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 2(xzbuf)
 // ind=0;
    j=0;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(k = 0; k < nz; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else{
          r+=0.1*xyBuf[i*ny];
      }         
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else  r += 0.1*rankA[RDX(i,j,k)];

          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else{
            r+=0.1*yzBuf[k];
          } 
          r += 0.1*rankA[RDX(i,j+1,k)];
          r+=0.1*xzBuf[i*nz+k];
 //         ind++;
          rankB[RDX(i,j,k)] = r;
    }
  }
   return 0;
}

int boundCalTask5(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz)
{
  int i,j,k;
 // int ind;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))

    //the first: data from 1(yzBuf)
   // ind=0;
    i=0;
    #pragma omp parallel for
    for(j = 0; j < ny; j ++) { 
      for(k = 0; k < nz; k ++) {
            double r = 0.4*rankA[RDX(i,j,k)];
      if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else{
          r+=0.1*xyBuf[j];
      }          
      if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
      else{
          r+=0.1*xzBuf[k];
      }         
      r += 0.1*rankA[RDX(i+1,j,k)];
      r += 0.1*yzBuf[j*nz+k];
 //     ind++;
      rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 6(xyBuf)
    //ind=0;
    k=nz-1;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(j = 0; j < ny; j ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else{
            r+=0.1*xzBuf[(i+1)*nz-1];
          }          
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else{
            r+=0.1*yzBuf[(j+1)*nz-1];
          }          
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else  r += 0.1*rankA[RDX(i,j,k)];
          r += 0.1*rankA[RDX(i,j,k-1)];
          r+=0.1*xyBuf[i*ny+j];
   //       ind++;

          rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 7(xzbuf)
 // ind=0;
    j=ny-1;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(k = 0; k < nz; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else{
          r+=0.1*xyBuf[(i+1)*ny-1];
      }

          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else{
            r+=0.1*yzBuf[(ny-1)*nz+k];
          }          
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else  r += 0.1*rankA[RDX(i,j,k)]; 
          r += 0.1*rankA[RDX(i,j-1,k)];
          r+=0.1*xzBuf[i*nz+k];
 //         ind++;
          rankB[RDX(i,j,k)] = r;
    }
  }
   return 0;
}
int boundCalTask6(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz)
{
  int i,j,k;
 // int ind;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))

    //the first: data from 2(yzBuf)
   // ind=0;
    i=0;
    #pragma omp parallel for
    for(j = 0; j < ny; j ++) { 
      for(k = 0; k < nz; k ++) {
            double r = 0.4*rankA[RDX(i,j,k)];
      if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else{
        r+=0.1*xyBuf[j];
      }         
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else r += 0.1*rankA[RDX(i,j,k)];      
      if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
      else{
          r+=0.1*xzBuf[k];
      }         
      r += 0.1*rankA[RDX(i+1,j,k)];
      r += 0.1*yzBuf[j*nz+k];
  //    ind++;
      rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 5(xyBuf)
   // ind=0;
    k=0;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(j = 0; j < ny; j ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else          r += 0.1*rankA[RDX(i,j,k)];
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else{
            r+=0.1*xzBuf[i*nz];
          }          
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else{
              r+=0.1*yzBuf[j*nz];
          }        
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else   r += 0.1*rankA[RDX(i,j,k)];
          r += 0.1*rankA[RDX(i,j,k+1)];
          r+=0.1*xyBuf[i*ny+j];
 //         ind++;

          rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 8(xzbuf)
 // ind=0;
    j=ny-1;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(k = 0; k < nz; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else{
        r+=0.1*xyBuf[(i+1)*ny-1];
      }          
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else  r += 0.1*rankA[RDX(i,j,k)];
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else{
              r+=0.1*yzBuf[(ny-1)*nz+k];
          }        
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else   r += 0.1*rankA[RDX(i,j,k)];
          r += 0.1*rankA[RDX(i,j-1,k)];
          r+=0.1*xzBuf[i*nz+k];
 //         ind++;
          rankB[RDX(i,j,k)] = r;
    }
  }
   return 0;
}
int boundCalTask7(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz)
{
  int i,j,k;
 // int ind;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))

    //the first: data from 3(yzBuf)
   // ind=0;
    i=0;
    #pragma omp parallel for
    for(j = 0; j < ny; j ++) { 
      for(k = 0; k < nz; k ++) {
            double r = 0.4*rankA[RDX(i,j,k)];
      if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else{
          r+=0.1*xyBuf[j];
      }          
      if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
      else{
        r+=0.1*xzBuf[k];
      }          
      if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
      else  r += 0.1*rankA[RDX(i,j,k)];      
      r += 0.1*rankA[RDX(i+1,j,k)];
      r += 0.1*yzBuf[j*nz+k];
 //     ind++;
      rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 8(xyBuf)
    //ind=0;
    k=nz-1;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(j = 0; j < ny; j ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else{
            r+=0.1*xzBuf[(i+1)*nz-1];
          }         
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else   r += 0.1*rankA[RDX(i,j,k)];      
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else{
            r+=0.1*yzBuf[(j+1)*nz-1];
          }          
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else  r += 0.1*rankA[RDX(i,j,k)];
          r += 0.1*rankA[RDX(i,j,k-1)];
          r+=0.1*xyBuf[i*ny+j];
   //       ind++;

          rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 5(xzbuf)
 // ind=0;
    j=0;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(k = 0; k < nz; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else          r += 0.1*rankA[RDX(i,j,k)];
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else{
          r+=0.1*xyBuf[i*ny];
      }

          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else {
              r+=0.1*yzBuf[k];
          }        
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else  r += 0.1*rankA[RDX(i,j,k)];
          r += 0.1*rankA[RDX(i,j+1,k)];
          r+=0.1*xzBuf[i*nz+k];
 //         ind++;
          rankB[RDX(i,j,k)] = r;
    }
  }
   return 0;
}
int boundCalTask8(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz)
{
  int i,j,k;
 // int ind;
  int xyInd,xzInd,yzInd;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))

    //the first: data from 5(yzBuf)
   // ind=0;
    i=0;
    #pragma omp parallel for
    for(j = 0; j < ny; j ++) { 
      for(k = 0; k < nz; k ++) {
            double r = 0.4*rankA[RDX(i,j,k)];
      if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else{
        r+=0.1*xyBuf[j];
      }          
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else  r += 0.1*rankA[RDX(i,j,k)];       
      if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
      else{
        r+=0.1*xzBuf[k];
      }          
      if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
      else  r += 0.1*rankA[RDX(i,j,k)];      
      r += 0.1*rankA[RDX(i+1,j,k)];
      r += 0.1*yzBuf[j*nz+k];
   //   ind++;
      rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 7(xyBuf)
    //ind=0;
    k=0;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(j = 0; j < ny; j ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(j !=  0)   r += 0.1*rankA[RDX(i,j-1,k)];
          else{
              r+=0.1*xzBuf[i*nz];
          }          
          if(j != ny-1) r += 0.1*rankA[RDX(i,j+1,k)];
          else  r += 0.1*rankA[RDX(i,j,k)];       
          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else{
            r+=0.1*yzBuf[j*nz]; 
          }          
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else  r += 0.1*rankA[RDX(i,j,k)];
          r += 0.1*rankA[RDX(i,j,k+1)];
          r+=0.1*xyBuf[i*ny+j];
  //        ind++;

         rankB[RDX(i,j,k)] = r;
    }
  }
  //data from 6(xzbuf)
 // ind=0;
    j=0;
    #pragma omp parallel for
    for(i = 0; i < nx; i ++) { 
      for(k = 0; k < nz; k ++) {
          double r = 0.4*rankA[RDX(i,j,k)];
          if(k !=  0)   r += 0.1*rankA[RDX(i,j,k-1)];
      else{
        r+=0.1*xyBuf[i*ny];
      }         
      if(k != nz-1) r += 0.1*rankA[RDX(i,j,k+1)];
      else   r += 0.1*rankA[RDX(i,j,k)];

          if(i !=  0)   r += 0.1*rankA[RDX(i-1,j,k)];
          else{
              r+=0.1*yzBuf[k];
          }          
          if(i != nx-1) r += 0.1*rankA[RDX(i+1,j,k)];
          else  r += 0.1*rankA[RDX(i,j,k)];
          r += 0.1*rankA[RDX(i,j+1,k)];
          r+=0.1*xzBuf[i*nz+k];
   //       ind++;
          rankB[RDX(i,j,k)] = r;
    }
  }
   return 0;
}

int boundCal(double *rankA,double *rankB,double *xyBuf, double *xzBuf,double *yzBuf,int nx,int ny,int nz,int myrank)
{
  switch(myrank)
  {
    case 1:
      boundCalTask1(rankA,rankB,xyBuf, xzBuf,yzBuf,nx,ny,nz);
      break;
    case 2:
      boundCalTask2(rankA,rankB,xyBuf, xzBuf,yzBuf,nx,ny,nz);
      break;
    case 3:
      boundCalTask3(rankA,rankB,xyBuf, xzBuf,yzBuf,nx,ny,nz);
      break;
    case 4:
      boundCalTask4(rankA,rankB,xyBuf, xzBuf,yzBuf,nx,ny,nz);
      break;
    case 5:
      boundCalTask5(rankA,rankB,xyBuf, xzBuf,yzBuf,nx,ny,nz);
      break;
    case 6:
      boundCalTask6(rankA,rankB,xyBuf, xzBuf,yzBuf,nx,ny,nz);
      break;
    case 7:
      boundCalTask7(rankA,rankB,xyBuf, xzBuf,yzBuf,nx,ny,nz);
      break;
    case 8:
      boundCalTask8(rankA,rankB,xyBuf, xzBuf,yzBuf,nx,ny,nz);
      break;
    default:
      break;
  }
  return 0;
}

//send buff
int sendBufTask(double *rankA,double *xySendBuf,double *xzSendBuf,double *yzSendBuf,int nx,int ny,int nz,int myrank)
{
  int i,j,k;
  int m_i,m_j,m_k;
  #define RDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))
  int ind;
  int flag1,flag2,flag3;
  m_i=0;  m_j=0; m_k=0;
  flag1=(myrank-1)/4; flag2=((myrank-1)%4)/2; flag3=(myrank-1)%2;
  if(flag1==0){
    m_i=nx-1;
  }
  if(flag2==0){
    m_j=ny-1;
  }
  if(flag3==0){
    m_k=nz-1;
  }

  k=m_k;
  #pragma omp parallel for
  for(i=0;i<nx;i++){
    for(j=0;j<ny;j++){
      xySendBuf[i*ny+j]=rankA[RDX(i,j,k)];
    }
  }

  i=m_i;
  #pragma omp parallel for
  for(j=0;j<ny;j++){
    for(k=0;k<nz;k++){
      yzSendBuf[j*nz+k]=rankA[RDX(i,j,k)];
    }
  }

  j=m_j;
  #pragma omp parallel for
  for(i=0;i<nx;i++){
    for(k=0;k<nz;k++){
      xzSendBuf[i*nz+k]=rankA[RDX(i,j,k)];
    }
  }
  return 0;
}


int main(int argc, char **argv) {
  double *A = NULL,*B=NULL,*partBuf = NULL;
  double *xyBuf=NULL,*xzBuf=NULL,*yzBuf=NULL,*xySendBuf=NULL,*yzSendBuf=NULL,*xzSendBuf=NULL;
  int myrank, nprocs, nx, ny, nz;
  int NX,NY,NZ,STEPS;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  NX = atoi(argv[1]), NY = atoi(argv[2]), NZ = atoi(argv[3]);
  STEPS = atoi(argv[4]);

  if(myrank == 0)
    printf("Size:%dx%dx%d, # of Steps: %d, # of procs: %d\n",     NX, NY, NZ, STEPS, nprocs);
  Info info = setup(NX, NY, NZ, nprocs);
  nx = info.nx, ny = info.ny, nz = info.nz;
  long long size = nx*ny*nz;
  A = (double*)malloc(size*sizeof(double));
  B = (double*)malloc(size*sizeof(double));
  Init(A, size);

  int xySize=nx*ny;
  int xzSize=nx*nz;
  int yzSize=ny*nz;
 
  struct timeval t1, t2;
  MPI_Barrier(MPI_COMM_WORLD), gettimeofday(&t1, NULL);
  MPI_Status status;

  xyBuf=(double*)malloc(xySize*sizeof(double));
  xzBuf=(double*)malloc(xzSize*sizeof(double));
  yzBuf=(double*)malloc(yzSize*sizeof(double));
  xySendBuf=(double*)malloc(xySize*sizeof(double));
  xzSendBuf=(double*)malloc(xzSize*sizeof(double));
  yzSendBuf=(double*)malloc(yzSize*sizeof(double));

  int s;
  int reqCount=6;
  MPI_Status m_status[6];  
  MPI_Request req[6]; 
  for(s=0;s<STEPS;s++){
        sendBufTask(A,xySendBuf,xzSendBuf,yzSendBuf,nx,ny,nz,myrank+1);
        switch(myrank){
          case 0:
          {
            MPI_Isend(&xySendBuf[0],xySize,MPI_DOUBLE,1,0,MPI_COMM_WORLD,&req[0]);
            MPI_Isend(&xzSendBuf[0],xzSize,MPI_DOUBLE,2,0,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(&yzSendBuf[0],yzSize,MPI_DOUBLE,4,0,MPI_COMM_WORLD,&req[2]);
            MPI_Irecv(&xyBuf[0],xySize,MPI_DOUBLE,1,1,MPI_COMM_WORLD,&req[3]);
            MPI_Irecv(&xzBuf[0],xzSize,MPI_DOUBLE,2,2,MPI_COMM_WORLD,&req[4]);
            MPI_Irecv(&yzBuf[0],yzSize,MPI_DOUBLE,4,4,MPI_COMM_WORLD,&req[5]);
            break;
          }
          case 1:
          {
            MPI_Isend(&xySendBuf[0],xySize,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&req[0]);
            MPI_Isend(&xzSendBuf[0],xzSize,MPI_DOUBLE,3,1,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(&yzSendBuf[0],yzSize,MPI_DOUBLE,5,1,MPI_COMM_WORLD,&req[2]);
            MPI_Irecv(&xyBuf[0],xySize,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&req[3]);
            MPI_Irecv(&xzBuf[0],xzSize,MPI_DOUBLE,3,3,MPI_COMM_WORLD,&req[4]);
            MPI_Irecv(&yzBuf[0],yzSize,MPI_DOUBLE,5,5,MPI_COMM_WORLD,&req[5]);
            break;
          }
          case 2:
          {
            MPI_Isend(&xySendBuf[0],xySize,MPI_DOUBLE,3,2,MPI_COMM_WORLD,&req[0]);
            MPI_Isend(&xzSendBuf[0],xzSize,MPI_DOUBLE,0,2,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(&yzSendBuf[0],yzSize,MPI_DOUBLE,6,2,MPI_COMM_WORLD,&req[2]);
            MPI_Irecv(&xyBuf[0],xySize,MPI_DOUBLE,3,3,MPI_COMM_WORLD,&req[3]);
            MPI_Irecv(&xzBuf[0],xzSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&req[4]);
            MPI_Irecv(&yzBuf[0],yzSize,MPI_DOUBLE,6,6,MPI_COMM_WORLD,&req[5]);
            break;
          }
          case 3:
          {
            MPI_Isend(&xySendBuf[0],xySize,MPI_DOUBLE,2,3,MPI_COMM_WORLD,&req[0]);
            MPI_Isend(&xzSendBuf[0],xzSize,MPI_DOUBLE,1,3,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(&yzSendBuf[0],yzSize,MPI_DOUBLE,7,3,MPI_COMM_WORLD,&req[2]);
            MPI_Irecv(&xyBuf[0],xySize,MPI_DOUBLE,2,2,MPI_COMM_WORLD,&req[3]);
            MPI_Irecv(&xzBuf[0],xzSize,MPI_DOUBLE,1,1,MPI_COMM_WORLD,&req[4]);
            MPI_Irecv(&yzBuf[0],yzSize,MPI_DOUBLE,7,7,MPI_COMM_WORLD,&req[5]);
            break;
          }
          case 4:
          {
            MPI_Isend(&xySendBuf[0],xySize,MPI_DOUBLE,5,4,MPI_COMM_WORLD,&req[0]);
            MPI_Isend(&xzSendBuf[0],xzSize,MPI_DOUBLE,6,4,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(&yzSendBuf[0],yzSize,MPI_DOUBLE,0,4,MPI_COMM_WORLD,&req[2]);
            MPI_Irecv(&xyBuf[0],xySize,MPI_DOUBLE,5,5,MPI_COMM_WORLD,&req[3]);
            MPI_Irecv(&xzBuf[0],xzSize,MPI_DOUBLE,6,6,MPI_COMM_WORLD,&req[4]);
            MPI_Irecv(&yzBuf[0],yzSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&req[5]);
            break;
          }
          case 5:
          {
            MPI_Isend(&xySendBuf[0],xySize,MPI_DOUBLE,4,5,MPI_COMM_WORLD,&req[0]);
            MPI_Isend(&xzSendBuf[0],xzSize,MPI_DOUBLE,7,5,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(&yzSendBuf[0],yzSize,MPI_DOUBLE,1,5,MPI_COMM_WORLD,&req[2]);
            MPI_Irecv(&xyBuf[0],xySize,MPI_DOUBLE,4,4,MPI_COMM_WORLD,&req[3]);
            MPI_Irecv(&xzBuf[0],xzSize,MPI_DOUBLE,7,7,MPI_COMM_WORLD,&req[4]);
            MPI_Irecv(&yzBuf[0],yzSize,MPI_DOUBLE,1,1,MPI_COMM_WORLD,&req[5]);
            break;
          }
          case 6:
          {
            MPI_Isend(&xySendBuf[0],xySize,MPI_DOUBLE,7,6,MPI_COMM_WORLD,&req[0]);
            MPI_Isend(&xzSendBuf[0],xzSize,MPI_DOUBLE,4,6,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(&yzSendBuf[0],yzSize,MPI_DOUBLE,2,6,MPI_COMM_WORLD,&req[2]);
            MPI_Irecv(&xyBuf[0],xySize,MPI_DOUBLE,7,7,MPI_COMM_WORLD,&req[3]);
            MPI_Irecv(&xzBuf[0],xzSize,MPI_DOUBLE,4,4,MPI_COMM_WORLD,&req[4]);
            MPI_Irecv(&yzBuf[0],yzSize,MPI_DOUBLE,2,2,MPI_COMM_WORLD,&req[5]);
            break;
          }
          case 7:
          {
            MPI_Isend(&xySendBuf[0],xySize,MPI_DOUBLE,6,7,MPI_COMM_WORLD,&req[0]);
            MPI_Isend(&xzSendBuf[0],xzSize,MPI_DOUBLE,5,7,MPI_COMM_WORLD,&req[1]);
            MPI_Isend(&yzSendBuf[0],yzSize,MPI_DOUBLE,3,7,MPI_COMM_WORLD,&req[2]);
            MPI_Irecv(&xyBuf[0],xySize,MPI_DOUBLE,6,6,MPI_COMM_WORLD,&req[3]);
            MPI_Irecv(&xzBuf[0],xzSize,MPI_DOUBLE,5,5,MPI_COMM_WORLD,&req[4]);
            MPI_Irecv(&yzBuf[0],yzSize,MPI_DOUBLE,3,3,MPI_COMM_WORLD,&req[5]);
            break;
          }
          default:
            break;
        }
         centerCal(A,B,nx,ny,nz,myrank+1); //center update
         MPI_Waitall(reqCount, req, m_status); 

         boundCal(A,B,xyBuf, xzBuf,yzBuf,nx,ny,nz,myrank+1);
         double *tmp=NULL;
         tmp=A,A=B,B=tmp;
  }

  //printf("myrank--->%d come here\n",myrank);
  MPI_Barrier(MPI_COMM_WORLD), gettimeofday(&t2, NULL);
  if(myrank == 0) printf("Total time: %.6lf\n", TIME(t1,t2));
  if(STEPS%2){  Check(B, size);}
  else{   Check(A, size);}
  if(myrank==0){
    free(A), free(B);
  }else{
    free(xyBuf);free(xzBuf),free(yzBuf); 
    free(xySendBuf);  free(xzSendBuf);  free(yzSendBuf);
  }

  MPI_Finalize();
}
















































