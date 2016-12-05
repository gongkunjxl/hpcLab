/**
 * 预处理文件: data.txt  第1: int nSize(矩阵中非零个数)
 *N个int值表示每行中第一个数值的偏移 nSize个列 nSize个double类型的值
 *v1_b.txt 下面N个double类型的值
 *v1_x0.txt下面N个double类型的值
 *
**/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 360
#define NY 180
#define NZ 38
#define N NX*NY*NZ
double *AV,*b,*x0;
int *AI,*AJ;
int nSize;

int InitAll()
{
    FILE *fp1=fopen("data.txt","rb");
    if(fp1==NULL){
        printf("can't open read file\n");
        exit(0);
    }
    int size;
    fread(&size,4,1,fp1);
    AV=(double *)malloc(size* sizeof(double));
    AI=(int*)malloc(N*sizeof(int));
    AJ=(int*)malloc(size* sizeof(int));
    fread(AI,4,N,fp1);
    fread(AJ,4,size,fp1);
    fread(AV,8,size,fp1);
    int p;
    /*for(p=0;p<30;p++)
    {
        printf("value-->%.30lf\n",AV[p]);
    }*/
    fclose(fp1); 	
    
    FILE *fp2=fopen("v1_b.txt","rb");
    if(fp2==NULL){
        printf("can't open read file\n");
        exit(0);
    }
    b=(double *)malloc(N* sizeof(double));
    fread(b,8,N,fp2);
    /*for(p=0;p<20;p++)
    {
        printf("value %d-->%.30lf\n",p,b[p]);
    }*/
    fclose(fp2);
 	
    FILE *fp3=fopen("v1_x0.txt","rb");
    if(fp3==NULL){
        printf("can't open read file\n");
        exit(0);
    }
    x0=(double *)malloc(N* sizeof(double));
    fread(x0,8,N,fp3);
    /*for(p=0;p<40;p++)
    {
        printf("value--%d----> %.30lf\n",p,x0[p]);
    }*/
    fclose(fp3);
    return size; 	
}

double checkSum()
{
  int ind=0;
  int i,j;
  int size;
  double tmp;
  double sum=0.0;
  for(i=0;i<N-1;i++) //N-1行
  {
    ind=AI[i];
    size=AI[i+1]-AI[i];
    tmp=0.0; 
    for(j=0;j<size;j++){
       tmp=tmp+AV[ind+j]*x0[AJ[ind+j]];	
    }
    sum=sum+(tmp-b[i])*(tmp-b[i]);
  }

  size=nSize-AI[N-1];
 printf("%d\n",AI[N-1]);
  tmp=0.0;
  ind=AI[N-1];
  for(j=0;j<size;j++){
    tmp=tmp+AV[ind+j]*x0[AJ[ind+j]];
  }
  sum=sum+(tmp-b[N-1])*(tmp-b[N-1]);
  
  return sum;
}



int main(int argc, char **argv) {
   AI=NULL;  AJ=NULL; AV=NULL; b=NULL; x0=NULL;
   printf(" start Init all data\n"); 
   nSize=InitAll();  
   printf(" end Init all data----N: %d   nSize: %d\n",N,nSize); 
   double checkValue=checkSum();
   printf("checksum ---->%e\n",checkValue); 

   free(AI); free(AJ);free(AV);free(b);free(x0);
   return 0;
}


































