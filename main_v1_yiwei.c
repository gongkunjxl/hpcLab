/**
 *一维切分
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
  Info result; 
  int myrank = 0;
  int nprocs=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if(myrank == 0)
  { 
    result.nx = NX/nprocs+NX%nprocs;
    result.ny = NY;
    result.nz = NZ;
  }
  else
  { 
    result.nx = NX/nprocs;
    result.ny = NY;
    result.nz = NZ;
  }
  return result;
}

//中心计算
int calGenenal(double *A,double *B,int nx,int ny,int nz,int myrank,int m_nprocs)
{
  #define IDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))
  int m_start=0;
  int m_end=0;
  if(myrank==0){
    m_start=0;
    m_end=nx-1;
  }
  else if(myrank==m_nprocs){
    m_start=1;
    m_end=nx;
  }else{
    m_start=1;
    m_end=nx-1;
  }
 
  int i, j, k;
 #pragma omp parallel for schedule (dynamic) 
 for (i = m_start; i < m_end; i++)
  {
    for(j = 0; j < ny; j ++) {
      for(k = 0; k < nz; k ++) {
        double r = 0.4*A[IDX(i,j,k)];
        if(k !=  0)   r += 0.1*A[IDX(i,j,k-1)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(k != nz-1) r += 0.1*A[IDX(i,j,k+1)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(j !=  0)   r += 0.1*A[IDX(i,j-1,k)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(j != ny-1) r += 0.1*A[IDX(i,j+1,k)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(i !=  0)   r += 0.1*A[IDX(i-1,j,k)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(i != nx-1) r += 0.1*A[IDX(i+1,j,k)];
        else          r += 0.1*A[IDX(i,j,k)];
        B[IDX(i,j,k)] = r;
      }
    }
  }
    return 0;
}

//边界计算
int boundCal(double *A,double *B,double *revBuff,int nx,int ny,int nz,int flag)
{
  int i, j, k;
  #define IDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))
  if(flag==1){
    i=0;
  }else{
    i=nx-1;
  }
  int ind=0;
  //#pragma omp parallel for shared(ind)
  for(j = 0; j < ny; j ++) {
    for(k = 0; k < nz; k ++) {
       double r = 0.4*A[IDX(i,j,k)];
        if(k !=  0)   r += 0.1*A[IDX(i,j,k-1)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(k != nz-1) r += 0.1*A[IDX(i,j,k+1)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(j !=  0)   r += 0.1*A[IDX(i,j-1,k)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(j != ny-1) r += 0.1*A[IDX(i,j+1,k)];
        else          r += 0.1*A[IDX(i,j,k)];
        if(flag==1)
        {
            r += 0.1*A[IDX(i+1,j,k)];
        }else{
          r += 0.1*A[IDX(i-1,j,k)];
        }
        r += 0.1*revBuff[ind];
        ind++;
        B[IDX(i,j,k)] = r;
    }
  }
  //printf("boud cal ----->%d\n",ind);
  return 0;
}

//边界数据
int sendData(double *A,double *sendBuff,int nx,int ny,int nz,int flag){

  int i, j, k;
  #define IDX(i,j,k) ((i)*ny*nz+(j)*nz+(k))
  if(flag==1){
    i=0;
  }else{
    i=nx-1;
  }
  int ind=0;
  //#pragma omp parallel for schedule(static)
  for(j = 0; j < ny; j ++) {
    for(k = 0; k < nz; k ++) {
        //sendBuff[j*nz+k]=A[IDX(i,j,k)];
        sendBuff[ind]=A[IDX(i,j,k)];
        ind++;
    }
  }
  return 0;
}


int main(int argc, char **argv) {
  double *A = NULL, *B = NULL;
  double *sendLeft=NULL,*sendRight=NULL,*revLeft=NULL,*revRight=NULL;
  int myrank, nprocs, nx, ny, nz;
  int NX,NY,NZ,STEPS;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  NX = atoi(argv[1]), NY = atoi(argv[2]), NZ = atoi(argv[3]);
  STEPS = atoi(argv[4]);

  if(myrank == 0)
    printf("Size:%dx%dx%d, # of Steps: %d, # of procs: %d\n",
        NX, NY, NZ, STEPS, nprocs);
  if(NX<NY){
    int tmp=NX;  NX=NY; NY=tmp;
  }
  if(NX<NZ){
     int tmp=NX;NX=NZ;NZ=tmp; 
  }
  if(NY>NZ){
    int tmp=NY; NY=NZ;NZ=tmp;
  }

  Info info = setup(NX, NY, NZ, nprocs);
  nx = info.nx, ny = info.ny, nz = info.nz;


  long long size = nx*ny*nz;
  A = (double*)malloc(size*sizeof(double));
  B = (double*)malloc(size*sizeof(double));
  Init(A, size);  

  struct timeval t1, t2;
  MPI_Barrier(MPI_COMM_WORLD), gettimeofday(&t1, NULL);
  int m_nprocs=nprocs-1;
  int count=ny*nz;
  if(myrank==0){
    sendRight=(double*)malloc(count*sizeof(double));
    revRight=(double*)malloc(count*sizeof(double));
  }
  else if(myrank==m_nprocs){
    sendLeft=(double*)malloc(count*sizeof(double));
    revLeft=(double*)malloc(count*sizeof(double));
  }
  else{
    sendRight=(double*)malloc(count*sizeof(double));
    revRight=(double*)malloc(count*sizeof(double));
    sendLeft=(double*)malloc(count*sizeof(double));
    revLeft=(double*)malloc(count*sizeof(double));
  }

  int s;
  MPI_Status m_status[4];
  MPI_Request m_req[4]; 
  MPI_Status mm_status[2];
  MPI_Request mm_req[2]; 
  for( s=0;s<STEPS;s++){
    if(myrank==0){
            
      sendData(A,sendRight,nx,ny,nz,2);
      MPI_Isend(&sendRight[0],count,MPI_DOUBLE,myrank+1,myrank,MPI_COMM_WORLD,&mm_req[0]);
      MPI_Irecv(&revRight[0],count,MPI_DOUBLE,myrank+1,myrank+1,MPI_COMM_WORLD,&mm_req[1]);
 
      calGenenal(A,B,nx,ny,nz,myrank,m_nprocs);
      MPI_Waitall(2,mm_req,mm_status);
      boundCal(A,B,revRight,nx,ny,nz,2);

    }
    else if(myrank==m_nprocs)
    {
      sendData(A,sendLeft,nx,ny,nz,1);
      MPI_Isend(&sendLeft[0],count,MPI_DOUBLE,myrank-1,myrank,MPI_COMM_WORLD,&mm_req[0]);
      MPI_Irecv(&revLeft[0],count,MPI_DOUBLE,myrank-1,myrank-1,MPI_COMM_WORLD,&mm_req[1]);
      
      calGenenal(A,B,nx,ny,nz,myrank,m_nprocs);
      MPI_Waitall(2,mm_req, mm_status);
      boundCal(A,B,revLeft,nx,ny,nz,1);  
    }
    else{ 
      sendData(A,sendLeft,nx,ny,nz,1);
      sendData(A,sendRight,nx,ny,nz,2);
   
      MPI_Isend(&sendRight[0],count,MPI_DOUBLE,myrank+1,myrank,MPI_COMM_WORLD,&m_req[0]);
      MPI_Isend(&sendLeft[0],count,MPI_DOUBLE,myrank-1,myrank,MPI_COMM_WORLD,&m_req[1]);
      MPI_Irecv(&revRight[0],count,MPI_DOUBLE,myrank+1,myrank+1,MPI_COMM_WORLD,&m_req[2]);
      MPI_Irecv(&revLeft[0],count,MPI_DOUBLE,myrank-1,myrank-1,MPI_COMM_WORLD,&m_req[3]);

      calGenenal(A,B,nx,ny,nz,myrank,m_nprocs);
      MPI_Waitall(4,m_req,m_status);
      boundCal(A,B,revLeft,nx,ny,nz,1);
      boundCal(A,B,revRight,nx,ny,nz,2);
    }

//    MPI_Barrier(MPI_COMM_WORLD);
    double *tmp = NULL;
    tmp=A,A=B,B=tmp;
}
  MPI_Barrier(MPI_COMM_WORLD), gettimeofday(&t2, NULL);
  if(myrank == 0) printf("Total time: %.6lf\n", TIME(t1,t2));
  if(STEPS%2) Check(B, size);
  else        Check(A, size);
  
  free(A),free(B);free(sendRight); free(sendLeft); free(revLeft); free(revRight);
  MPI_Finalize();
}






































