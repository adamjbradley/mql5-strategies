//--- by default some GPU doesn't support doubles
//--- cl_khr_fp64 directive is used to enable work with doubles
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForward(__global double *matrix_w,
                              __global double *matrix_i,
                              __global double *matrix_o,
                              int inputs, int activation)
  {
   int i=get_global_id(0);
   double sum=0.0;
   double4 inp, weight;
   int shift=(inputs+1)*i;
   for(int k=0; k<=inputs; k=k+4)
     {
      switch(inputs-k)
        {
         case 0:
           inp=(double4)(1,0,0,0);
           weight=(double4)(matrix_w[shift+k],0,0,0);
           break;
         case 1:
           inp=(double4)(matrix_i[k],1,0,0);
           weight=(double4)(matrix_w[shift+k],matrix_w[shift+k+1],0,0);
           break;
         case 2:
           inp=(double4)(matrix_i[k],matrix_i[k+1],1,0);
           weight=(double4)(matrix_w[shift+k],matrix_w[shift+k+1],matrix_w[shift+k+2],0);
           break;
         case 3:
           inp=(double4)(matrix_i[k],matrix_i[k+1],matrix_i[k+2],1);
           weight=(double4)(matrix_w[shift+k],matrix_w[shift+k+1],matrix_w[shift+k+2],matrix_w[shift+k+3]);
           break;
         default:
           inp=(double4)(matrix_i[k],matrix_i[k+1],matrix_i[k+2],matrix_i[k+3]);
           weight=(double4)(matrix_w[shift+k],matrix_w[shift+k+1],matrix_w[shift+k+2],matrix_w[shift+k+3]);
           break;
        }
      sum+=dot(inp,weight);
     }
   switch(activation)
     {
      case 0:
        sum=tanh(sum);
        break;
      case 1:
        sum=pow((1+exp(-sum)),-1);
        break;
     }
   matrix_o[i]=sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CaclOutputGradient(__global double *matrix_t,
                                 __global double *matrix_o,
                                 __global double *matrix_ig,
                                 int activation)
  {
   int i=get_global_id(0);
   double temp=0;
   double out=matrix_o[i];
   switch(activation)
     {
      case 0:
        temp=clamp(matrix_t[i],-1.0,1.0)-out;
        temp=temp*(1+out)*(1-(out==1 ? 0.99 : out));
        break;
      case 1:
        temp=clamp(matrix_t[i],0.0,1.0)-out;
        temp=temp*(out==0 ? 0.01 : out)*(1-(out==1 ? 0.99 : out));
        break;
     }
   matrix_ig[i]=temp;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CaclHiddenGradient(__global double *matrix_w,
                              __global double *matrix_g,
                              __global double *matrix_o,
                              __global double *matrix_ig,
                              int outputs, int activation)
  {
   int i=get_global_id(0);
   double sum=0;
   double out=matrix_o[i];
   double4 grad, weight;
   int shift=(outputs+1)*i;
   for(int k=0;k<outputs;k+=4)
     {
      switch(outputs-k)
        {
         case 0:
           grad=(double4)(1,0,0,0);
           weight=(double4)(matrix_w[shift+k],0,0,0);
           break;
         case 1:
           grad=(double4)(matrix_g[k],1,0,0);
           weight=(double4)(matrix_w[shift+k],matrix_w[shift+k+1],0,0);
           break;
         case 2:
           grad=(double4)(matrix_g[k],matrix_g[k+1],1,0);
           weight=(double4)(matrix_w[shift+k],matrix_w[shift+k+1],matrix_w[shift+k+2],0);
           break;
         case 3:
           grad=(double4)(matrix_g[k],matrix_g[k+1],matrix_g[k+2],1);
           weight=(double4)(matrix_w[shift+k],matrix_w[shift+k+1],matrix_w[shift+k+2],matrix_w[shift+k+3]);
           break;
         default:
           grad=(double4)(matrix_g[k],matrix_g[k+1],matrix_g[k+2],matrix_g[k+3]);
           weight=(double4)(matrix_w[shift+k],matrix_w[shift+k+1],matrix_w[shift+k+2],matrix_w[shift+k+3]);
           break;
        }
      sum+=dot(grad,weight);
     }
   switch(activation)
     {
      case 0:
        sum=clamp(sum+out,-1.0,1.0);
        sum=(sum-out)*(1+out)*(1-(out==1 ? 0.99 : out));
        break;
      case 1:
        sum=clamp(sum+out,0.0,1.0);
        sum=(sum-out)*(out==0 ? 0.01 : out)*(1-(out==1 ? 0.99 : out));
        break;
     }
   matrix_ig[i]=sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UpdateWeights(__global double *matrix_w,
                                __global double *matrix_g,
                                __global double *matrix_i,
                                __global double *matrix_dw,
                                int inputs, double learning_rates, double momentum)
  {
   int i=get_global_id(0);
   int j=get_global_id(1);
   int wi=i*(inputs+1)+j; 
   double delta=learning_rates*matrix_g[i]*(j<inputs ? matrix_i[j] : 1) + momentum*matrix_dw[wi];
   matrix_dw[wi]=delta;
   matrix_w[wi]+=delta;
  };
//+------------------------------------------------------------------+