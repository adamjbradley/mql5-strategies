/// \file
/// \brief NeuroNet.cl
/// Library consist OpenCL kernels
/// \author <A HREF="https://www.mql5.com/en/users/dng"> DNG </A>
/// \copyright Copyright 2019, DNG
//---
//--- by default some GPU doesn't support floats
//--- cl_khr_fp64 directive is used to enable work with floats
// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define l1 1.0e-3f
#define l2 1.0e-3f
#define MAX_WEIGHT 1.0e-3f
#define MAX_GRAD 1.0e-3f
#define LOCAL_ARRAY_SIZE 64
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float Activation(float value, int function)
  {
   if(isnan(value) || isinf(value))
      return 0;
//---
   float result = value;
   switch(function)
     {
      case 0:
         result = tanh(clamp(value, -20.0f, 20.0f));
         break;
      case 1:  //Sigmoid
         result = 1 / (1 + exp(clamp(-value, -20.0f, 20.0f)));
         break;
      case 2:  //LReLU
         if(value < 0)
            result *= 0.01f;
         break;
      case 3:  //SoftPlus
         result = (value >= 20.0f ? 1.0f : (value <= -20.0f ? 0.0f : log(1 + exp(value))));
         break;
      default:
         break;
     }
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float Deactivation(float grad, float inp_value, int function)
  {
   float result = grad;
//---
   if(isnan(inp_value) || isinf(inp_value) ||
      isnan(grad) || isinf(grad))
      result = 0;
   else
      switch(function)
        {
         case 0: //TANH
            result = clamp(grad + inp_value, -1.0f, 1.0f) - inp_value;
            result *= 1.0f - pow(inp_value, 2.0f);
            break;
         case 1:  //Sigmoid
            result = clamp(grad + inp_value, 0.0f, 1.0f) - inp_value;
            result *= inp_value * (1.0f - inp_value);
            break;
         case 2: //LReLU
            if(inp_value < 0)
               result *= 0.01f;
            break;
         case 3:  //SoftPlus
            result *= Activation(inp_value, 1);
            break;
         default:
            break;
        }
//---
   return clamp(result, -MAX_GRAD, MAX_GRAD);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_ff Feed forward process kernel
/// Describes the forward path process for the Neuron Base (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8435#para41">the link.</A>
//+------------------------------------------------------------------+
__kernel void FeedForward(__global float *matrix_w, ///<[in] Weights matrix (m+1)*n, where m - number
                          ///< of neurons in layer and n - number of outputs
                          ///< (neurons in next layer)
                          __global float *matrix_i, ///<[in] Inputs tensor
                          __global float *matrix_o, ///<[out] Output tensor
                          int inputs,               ///< Number of inputs
                          int activation            ///< Activation type (#ENUM_ACTIVATION)
                         )
  {
   int i = get_global_id(0);
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs + 1) * i;
   for(int k = 0; k <= inputs; k = k + 4)
     {
      switch(inputs - k)
        {
         case 0:
            inp = (float4)(1, 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 1:
            inp = (float4)(matrix_i[k], 1, 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i[k], matrix_i[k + 1], 1, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], 0);
            break;
         case 3:
            inp = (float4)(matrix_i[k], matrix_i[k + 1], matrix_i[k + 2], 1);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
         default:
            inp = (float4)(matrix_i[k], matrix_i[k + 1], matrix_i[k + 2],
                           matrix_i[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
   if(isnan(sum) || isinf(sum))
      sum = 0;
   matrix_o[i] = Activation(sum, activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_gr  Neuron Base Output Gradients Calculation kernel
/// Describes the process of output gradients calculation for the Neuron Base
/// (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8435#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcOutputGradient(__global float *matrix_t,  ///<[in] Target tensor
                                 __global float *matrix_o,  ///<[in] Output tensor
                                 __global float *matrix_ig, ///<[out] Tensor of gradients
                                 int activation, ///< Activation type (#ENUM_ACTIVATION)
                                 float error)
  {
   int i = get_global_id(0);
   float out = matrix_o[i];
   float temp = 0;
   if(!isnan(out) && !isinf(out))
      switch(activation)
        {
         case 0:
            // temp=clamp(matrix_t[i],-1.0,1.0)-out;
            temp = 2.0f * (matrix_t[i] - out);
            break;
         case 1:
            // temp=clamp(matrix_t[i],0.0,1.0)-out;
            temp = 2 * (matrix_t[i] - out) * error;
            temp = temp * out * (1 - out);
            break;
         case 2:
            // temp=(matrix_t[i]-out)*(out>=0 ? 1.0 : 0.01);
            temp = (2 * (matrix_t[i] - out) * error) * (out >= 0 ? 1.0f : 0.01f);
            break;
         default:
            temp = 2 * (matrix_t[i] - out) * error;
            break;
        }
   matrix_ig[i] = temp;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_gr  Neuron Base Hidden Gradients Calculation kernel
/// Describes the process of hidden gradients calculation for the Neuron Base
/// (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8435#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradient(__global float *matrix_w,  ///<[in] Weights matrix (m+1)*n, where m - number
                                 ///< of neurons in previous layer and n - number
                                 ///< of neurons in current layer
                                 __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                 __global float *matrix_o,  ///<[in] Previous layer Output tensor
                                 __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
                                 int outputs,               ///< Number of outputs
                                 int activation             ///< Activation type (#ENUM_ACTIVATION)
                                )
  {
   int i = get_global_id(0);
   int inputs = get_global_size(0);
   float sum = 0;
   float out = matrix_o[i];
   float4 grad, weight;
   for(int k = 0; k < outputs; k += 4)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[k], 0, 0, 0);
            break;
         case 2:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], 0, 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2], 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i],
                              matrix_w[(k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2],
                            matrix_g[k + 3]);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i],
                              matrix_w[(k + 2) * (inputs + 1) + i],
                              matrix_w[(k + 3) * (inputs + 1) + i]);
            break;
        }
      //---
      if(isnan(weight.s0) || isinf(weight.s0))
         weight.s0 = 0;
      if(isnan(weight.s1) || isinf(weight.s1))
         weight.s1 = 0;
      if(isnan(weight.s2) || isinf(weight.s2))
         weight.s2 = 0;
      if(isnan(weight.s3) || isinf(weight.s3))
         weight.s3 = 0;
      if(isnan(grad.s0) || isinf(grad.s0))
         grad.s0 = 0;
      if(isnan(grad.s1) || isinf(grad.s1))
         grad.s1 = 0;
      if(isnan(grad.s2) || isinf(grad.s2))
         grad.s2 = 0;
      if(isnan(grad.s3) || isinf(grad.s3))
         grad.s3 = 0;
      //---
      sum += dot(grad, weight);
     }
//---
   matrix_ig[i] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base SGD Updating Weights Calculation kernel
/// Describes the process of SGD optimization weights for the Neuron Base
/// (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8435#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateWeightsMomentum(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
                                    ///< number of neurons in previous layer and n -
                                    ///< number of neurons in current layer
                                    __global float *matrix_g, ///<[in] Tensor of gradients at current layer
                                    __global float *matrix_i, ///<[in] Inputs tensor
                                    __global float *matrix_dw, ///<[in,out] Matrix of delta weights in last correction
                                    int inputs,     ///< Number of inputs
                                    float learning_rates, ///< Learning rates
                                    float momentum        ///< Momentum multiplier
                                   )
  {
   int i = get_global_id(0);
   int j = get_global_id(1);
   int wi = i * (inputs + 1) + j;
   float delta = learning_rates * matrix_g[i] * (j < inputs ? matrix_i[j] : 1) +
                 momentum * matrix_dw[wi];
   if(!isnan(delta) || !isinf(delta))
     {
      matrix_dw[wi] = delta;
      if(fabs(delta) > 0)
         matrix_w[wi] = clamp(matrix_w[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base Adam Updating Weights Calculation
/// kernel
/// Describes the process of Adam optimization weights for the Neuron Base
/// (#CNeuronBaseOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8598#para31">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdam(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
                                ///< number of neurons in previous layer and n -
                                ///< number of neurons in current layer
                                __global const float
                                *matrix_g, ///<[in] Tensor of gradients at current layer
                                __global const float *matrix_i, ///<[in] Inputs tensor
                                __global float *matrix_m,       ///<[in,out] Matrix of first momentum
                                __global float *matrix_v,       ///<[in,out] Matrix of seconfd momentum
                                const int inputs,               ///< Number of inputs
                                const float l,                  ///< Learning rates
                                const float b1,                 ///< First momentum multiplier
                                const float b2                  ///< Second momentum multiplier
                               )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = i * (inputs + 1) + j;
   float m, v, weight, inp;
   inp = (j == inputs ? 1.0f : matrix_i[j]);
   weight = matrix_w[wi];
   m = matrix_m[wi];
   v = matrix_v[wi];
//---
   if(isnan(inp) || isinf(inp))
      inp = 0;
//---
   float g = matrix_g[i] * inp;
   float mt = b1 * m + (1 - b1) * g;
   float vt = b2 * v + (1 - b2) * pow(g, 2);
   float delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      matrix_w[wi] = clamp(matrix_w[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[wi] = mt;
   matrix_v[wi] = vt;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_base_opt  Neuron Base Least Squares Updating Weights
/// Calculation kernel
/// Describes the process of Least Squares optimization weights for the Neuron
/// Base (#CNeuronBaseOCL).
//\details Detailed description on <A
// HREF="https://www.mql5.com/ru/articles/8598#para31">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateWeightsLS(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
                              ///< number of neurons in previous layer and n -
                              ///< number of neurons in current layer
                              __global const float
                              *matrix_g, ///<[in] Tensor of gradients at current layer
                              __global const float *matrix_i, ///<[in] Inputs tensor
                              __global float *matrix_xg,      ///<[in,out] Matrix of summ x*g
                              __global float *matrix_xx,      ///<[in,out] Matrix of summ x*x
                              const int inputs,               ///< Number of inputs
                              const float l,                  ///< Learning rates
                              const int update                ///< Update flag
                             )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = i * (inputs + 1) + j * 4;
   float4 xg, xx, weight, inp;
   switch(inputs + 1 - j * 4)
     {
      case 0:
         inp = (float4)(1, 0, 0, 0);
         weight = (float4)(matrix_w[wi], 0, 0, 0);
         break;
      case 1:
         inp = (float4)(matrix_i[j * 4], 1, 0, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], 0, 0);
         break;
      case 2:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], 1, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], 0);
         break;
      case 3:
         inp =
            (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], matrix_i[j * 4 + 2], 1);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2],
                           matrix_w[wi + 3]);
         break;
      default:
         inp = (float4)(matrix_i[j * 4], matrix_i[j * 4 + 1], matrix_i[j * 4 + 2],
                        matrix_i[j * 4 + 3]);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2],
                           matrix_w[wi + 3]);
         break;
     }
   xg = (float4)(matrix_g[i]) * inp;
   xx = pow(inp, 2.0f);
   switch(min(inputs + 1 - j * 4, 3))
     {
      case 3:
         if(update)
           {
            matrix_w[wi + 3] =
               matrix_w[wi + 3] + l * (matrix_xg[wi + 3] + xg.s3) /
               (matrix_xx[wi + 3] + xx.s3 + 1.0e-37f);
            matrix_xg[wi + 3] = 0;
            matrix_xx[wi + 3] = 0;
           }
         else
           {
            matrix_xg[wi + 3] += xg.s3;
            matrix_xx[wi + 3] += xx.s3;
           }
      case 2:
         if(update)
           {
            matrix_w[wi + 2] =
               matrix_w[wi + 2] + l * (matrix_xg[wi + 2] + xg.s2) /
               (matrix_xx[wi + 2] + xx.s2 + 1.0e-37f);
            matrix_xg[wi + 2] = 0;
            matrix_xx[wi + 2] = 0;
           }
         else
           {
            matrix_xg[wi + 2] += xg.s2;
            matrix_xx[wi + 2] += xx.s2;
           }
      case 1:
         if(update)
           {
            matrix_w[wi + 1] =
               matrix_w[wi + 1] + l * (matrix_xg[wi + 1] + xg.s1) /
               (matrix_xx[wi + 1] + xx.s1 + 1.0e-37f);
            matrix_xg[wi + 1] = 0;
            matrix_xx[wi + 1] = 0;
           }
         else
           {
            matrix_xg[wi + 1] += xg.s1;
            matrix_xx[wi + 1] += xx.s1;
           }
      case 0:
         if(update)
           {
            matrix_w[wi] = matrix_w[wi] + l * (matrix_xg[wi] + xg.s0) /
                           (matrix_xx[wi] + xx.s0 + 1.0e-37f);
            matrix_xg[wi] = 0;
            matrix_xx[wi] = 0;
           }
         else
           {
            matrix_xg[wi] += xg.s0;
            matrix_xx[wi] += xx.s0;
           }
         break;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_proof_ff
/// Kernel of the Pooling neuron for Feed forward process (#CNeuronProofOCL)
//+------------------------------------------------------------------+
__kernel void FeedForwardProof(__global float *matrix_i, ///<[in] Inputs tensor
                               __global float *matrix_o, ///<[out] Output tensor
                               int inputs,               ///< Number of inputs
                               int window, ///< Size of input window
                               int step    ///< Step size
                              )
  {
   int i = get_global_id(0);
   int pos = i * step;
   float result = matrix_i[pos];
   for(int k = 1; k < window; k = k + 1)
     {
      int shift = k + pos;
      if(shift >= inputs)
         break;
      result = max(result, matrix_i[shift]);
     }
   matrix_o[i] = result;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_proof_gr
/// Kernel of the Pooling neuron to transfer gradient to previous layer
/// (#CNeuronProofOCL)
//+------------------------------------------------------------------+
__kernel void CalcInputGradientProof(__global float *matrix_i,  ///<[in] Inputs tensor
                                     __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_o,  ///<[in] Output tensor
                                     __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
                                     int outputs,               ///< Number of outputs
                                     int window,                ///< Size of input window
                                     int step                   ///< Step size
                                    )
  {
   int i = get_global_id(0);
   float prev_gradient = 0;
   float value = matrix_i[i];
   int start = i - window + step;
   start = (start - start % step) / step;
   int stop = (i - i % step) / step + 1;
   for(int out = max(0, start); out < min(outputs, stop); out++)
     {
      if(value == matrix_o[out])
         prev_gradient += matrix_g[out];
     }
   matrix_ig[i] = prev_gradient;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_ff
/// Kernel of the Convolution neuron for Feed forward process (#CNeuronConvOCL)
//+------------------------------------------------------------------+
__kernel void FeedForwardConv(__global float *matrix_w, ///<[in] Weights matrix (m+1)*n, where m - input
                              ///< window and n - output window
                              __global float *matrix_i, ///<[in] Inputs tensor
                              __global float *matrix_o, ///<[out] Output tensor
                              const int inputs,         ///< Number of inputs
                              const int step,           ///< Step size
                              const int window_in,      ///< Size of input window
                              const int window_out,     ///< Size of output window
                              const int activation      ///< Activation type (#ENUM_ACTIVATION)
                             )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t outputs = get_global_size(0);
//---
   const int shift_out = window_out * i;
   const int shift_in = step * i;
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * window_out * outputs;
   const int shift_var_w = v * window_out * (window_in + 1);
//---
   float sum = 0;
   float4 inp, weight;
//---
   for(int out = 0; out < window_out; out++)
     {
      int shift = (window_in + 1) * out;
      int stop = (window_in <= (inputs - shift_in) ? window_in : (inputs - shift_in));
      for(int k = 0; k <= stop; k += 4)
        {
         switch(stop - k)
           {
            case 0:
               inp = (float4)(1, 0, 0, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + k], 0, 0, 0);
               break;
            case 1:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], 1, 0, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1], 0, 0);
               break;
            case 2:
               inp =
                  (float4)(matrix_i[shift_var_in + shift_in + k], matrix_i[shift_var_in + shift_in + k + 1], 1, 0);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + k + 2], 0);
               break;
            case 3:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], matrix_i[shift_var_in + shift_in + k + 1],
                              matrix_i[shift_var_in + shift_in + k + 2], 1);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + k + 2], matrix_w[shift_var_w + shift + k + 3]);
               break;
            default:
               inp = (float4)(matrix_i[shift_var_in + shift_in + k], matrix_i[shift_var_in + shift_in + k + 1],
                              matrix_i[shift_var_in + shift_in + k + 2], matrix_i[shift_var_in + shift_in + k + 3]);
               weight = (float4)(matrix_w[shift_var_w + shift + k], matrix_w[shift_var_w + shift + k + 1],
                                 matrix_w[shift_var_w + shift + k + 2], matrix_w[shift_var_w + shift + k + 3]);
               break;
           }
         sum += dot(inp, weight);
        }
      if(isnan(sum))
         sum = 0;
      //---
      matrix_o[shift_var_out + out + shift_out] = Activation(sum, activation);;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_gr
/// Kernel of the Convolution neuron to transfer gradient to previous layer
/// (#CNeuronConvOCL)
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientConv(__global float *matrix_w,     ///<[in] Weights matrix (m+1)*n, where m - input
                                     ///< window and n - output window
                                     __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_o,  ///<[in] Output tensor
                                     __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
                                     const int outputs,               ///< Number of outputs
                                     const int step,                  ///< Step size
                                     const int window_in,             ///< Size of input window
                                     const int window_out,            ///< Size of output window
                                     const int activation,            ///< Activation type (#ENUM_ACTIVATION)
                                     const int shift_out              ///< Shift in output and gradient buffer
                                    )
  {
   const size_t i = get_global_id(0);
   const size_t inputs = get_global_size(0);
   const size_t v = get_global_id(1);
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * outputs;
   const int shift_var_w = v * window_out * (window_in + 1);
//---
   float sum = 0;
   float out = matrix_o[shift_var_in + i];
   const int start = max((start + step - 1) / step, 0);
   int stop = (i + step - 1) / step + 1;
   if(stop > (outputs / window_out))
      stop = outputs / window_out;
   for(int h = 0; h < window_out; h ++)
     {
      for(int k = start; k < stop; k++)
        {
         int shift_g = k * window_out + h;
         int shift_w = (stop - k - 1) * step + i % step + h * (window_in + 1);
         if(shift_g >= outputs || shift_w >= (window_in + 1) * window_out)
            break;
         float grad = matrix_g[shift_out + shift_g + shift_var_out];
         sum += grad * matrix_w[shift_w + shift_var_w];
        }
     }
//---
   matrix_ig[shift_var_in + i] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron SGD optimization Updating Weights
/// Calculation kernel
/// Describes the process of SGD optimization weights for the Convolution Neuron
/// (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvMomentum(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                        ///< input window and n - output window
                                        __global float *matrix_g, ///<[in] Tensor of gradients at current layer
                                        __global float *matrix_i, ///<[in] Inputs tensor
                                        __global float
                                        *matrix_dw, ///<[in,out] Matrix of delta weights in last correction
                                        int inputs,     ///< Number of inputs
                                        float learning_rates, ///< Learning rates
                                        float momentum,       ///< Momentum multiplier
                                        int window_in,        ///< Size of input window
                                        int window_out,       ///< Size of output window
                                        int step              ///< Step size
                                       )
  {
   const size_t i = get_global_id(0);
//---
   const int v = i / ((window_in + 1) * window_out);
   const int shift = i % (window_in + 1);
   const int shift_out = i / (window_in + 1) - v;
   const int total = (inputs - window_in + step - 1) / step;
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * total * window_out;
//---
   float grad = 0;
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in && (shift + t * window_in) >= inputs)
         break;
      grad += matrix_g[t * window_out + shift_out + shift_var_out] *
              (shift == window_in ? 1 : matrix_i[shift + t * step + shift_var_in]);
     }
   float delta = learning_rates * grad + momentum * matrix_dw[i];
   if(!isnan(delta))
     {
      matrix_dw[i] = delta;
      if(fabs(delta) > 0)
         matrix_w[i] = clamp(matrix_w[i] + delta, -MAX_WEIGHT, MAX_WEIGHT);
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron Adam optimization Updating
/// Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Convolution
/// Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvAdam(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                    ///< input window and n - output window
                                    __global const float *matrix_g, ///<[in] Tensor of gradients at current layer
                                    __global const float *matrix_i, ///<[in] Inputs tensor
                                    __global float *matrix_m,       ///<[in] Matrix of first momentum
                                    __global float *matrix_v,       ///<[in] Matrix of seconfd momentum
                                    const int inputs,               ///< Number of inputs
                                    const float l,                  ///< Learning rates
                                    const float b1,                 ///< First momentum multiplier
                                    const float b2,                 ///< Second momentum multiplier
                                    int window_in,                  ///< Size of input window
                                    int window_out,                 ///< Size of output window
                                    int step                        ///< Step size
                                   )
  {
   const size_t i = get_global_id(0);
//---
   const int v = i / ((window_in + 1) * window_out);
   const int shift = i % (window_in + 1);
   const int shift_out = i / (window_in + 1) - v;
   const int total = (inputs - window_in + step - 1) / step;
//---
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * total * window_out;
//---
   float grad = 0;
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in && (shift + t * window_in) >= inputs)
         break;
      grad += matrix_g[t * window_out + shift_out + shift_var_out] *
              (shift == window_in ? 1 : matrix_i[shift + t * step + shift_var_in]);
     }
   float mt = clamp(b1 * matrix_m[i] + (1 - b1) * grad, -1.0e5f, 1.0e5f);
   if(isnan(mt) || isinf(mt))
      mt = 0;
   float vt = clamp(b2 * matrix_v[i] + (1 - b2) * pow(grad, 2), 1.0e-6f, 1.0e6f);
   if(isnan(vt) || isinf(vt))
      vt = 1.0e-6f;
   float weight = clamp(matrix_w[i] + l * mt / sqrt(vt), -MAX_WEIGHT, MAX_WEIGHT);
   matrix_w[i] = weight;
   matrix_m[i] = mt;
   matrix_v[i] = vt;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_conv_opt Convolution Neuron Least Squares optimization
/// Updating Weights Calculation kernel
/// Describes the process of Least Squares optimization weights for the
/// Convolution Neuron (#CNeuronConvOCL).
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvLS(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                  ///< input window and n - output window
                                  __global const float
                                  *matrix_g, ///<[in] Tensor of gradients at current layer
                                  __global const float *matrix_i, ///<[in] Inputs tensor
                                  __global float *matrix_xg,      ///<[in] Matrix of summ x*g
                                  __global float *matrix_xx,      ///<[in] Matrix of summ x*x
                                  const int inputs,               ///< Number of inputs
                                  const float l,                  ///< Learning rates
                                  const int update,               ///< Update flag
                                  int window_in,                  ///< Size of input window
                                  int window_out,                 ///< Size of output window
                                  int step                        ///< Step size
                                 )
  {
   const int i = get_global_id(0);
   if(i > window_in)
      return;
//---
   int total = (inputs - (window_in - step)) % step;
   total = (inputs - (window_in - step) - total) / step + (total > 0 ? 1 : 0);
   for(int out = 0; out < window_out; out++)
     {
      if((window_out - out) > 4)
        {
         float4 xg = {0, 0, 0, 0};
         float x2 = 0;
         int shift_w = i + out * (window_in + 1);
         for(int t = 0; t < total; t++)
           {
            if(i != window_in && (i + t * window_in) >= inputs)
               break;
            xg += (float4)(matrix_g[t * window_out + out],
                           matrix_g[t * window_out + out + 1],
                           matrix_g[t * window_out + out + 2],
                           matrix_g[t * window_out + out + 3]) *
                  (i == window_in ? 1 : matrix_i[i + t * step]);
            x2 += (i == window_in ? 1 : pow(matrix_i[i + t * step], 2.0f));
           }
         if(update)
           {
            xg = (float4)(matrix_xg[shift_w], matrix_xg[shift_w + window_in + 1],
                          matrix_xg[shift_w + 2 * (window_in + 1)],
                          matrix_xg[shift_w + 3 * (window_in + 1)]) +
                 xg;
            float4 xx =
               (float4)(matrix_xx[shift_w], matrix_xx[shift_w + window_in + 1],
                        matrix_xx[shift_w + 2 * (window_in + 1)],
                        matrix_xx[shift_w + 3 * (window_in + 1)]) +
               x2;
            float4 delta = l * xg / (xx + 1.0e-37f);
            float4 weight =
               (float4)(matrix_w[shift_w], matrix_w[shift_w + (window_in + 1)],
                        matrix_w[shift_w + 2 * (window_in + 1)],
                        matrix_w[shift_w + 3 * (window_in + 1)]) +
               delta;
            matrix_w[shift_w] = weight.s0;
            matrix_w[shift_w + (window_in + 1)] = weight.s1;
            matrix_w[shift_w + 2 * (window_in + 1)] = weight.s2;
            matrix_w[shift_w + 3 * (window_in + 1)] = weight.s3;
            matrix_xg[shift_w] = 0;
            matrix_xg[shift_w + (window_in + 1)] = 0;
            matrix_xg[shift_w + 2 * (window_in + 1)] = 0;
            matrix_xg[shift_w + 3 * (window_in + 1)] = 0;
            matrix_xx[shift_w] = 0;
            matrix_xx[shift_w + (window_in + 1)] = 0;
            matrix_xx[shift_w + 2 * (window_in + 1)] = 0;
            matrix_xx[shift_w + 3 * (window_in + 1)] = 0;
           }
         else
           {
            matrix_xg[shift_w] += xg.s0;
            matrix_xg[shift_w + (window_in + 1)] += xg.s1;
            matrix_xg[shift_w + 2 * (window_in + 1)] += xg.s2;
            matrix_xg[shift_w + 3 * (window_in + 1)] += xg.s3;
            matrix_xx[shift_w] = matrix_xx[shift_w + (window_in + 1)] =
                                    matrix_xx[shift_w + 2 * (window_in + 1)] =
                                       matrix_xx[shift_w + 3 * (window_in + 1)] += x2;
           }
         out += 3;
        }
      else
        {
         float xg = 0;
         float xx = 0;
         int shift_w = i + out * (window_in + 1);
         for(int t = 0; t < total; t++)
           {
            if(i != window_in && (i + t * window_in) >= inputs)
               break;
            xg += matrix_g[t * window_out + out] *
                  (i == window_in ? 1 : matrix_i[i + t * step]);
            xx += (i == window_in ? 1 : pow(matrix_i[i + t * step], 2.0f));
           }
         if(update)
           {
            xg = matrix_xg[shift_w] + xg;
            xx = matrix_xx[shift_w] + xx;
            float delta = l * xg / (xx + 1.0e-37f);
            matrix_w[shift_w] = matrix_w[shift_w] + delta;
            matrix_xg[shift_w] = 0;
            matrix_xx[shift_w] = 0;
           }
         else
           {
            matrix_xg[shift_w] += xg;
            matrix_xx[shift_w] += xx;
           }
        }
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Attention Neuron Score calculation kernel |
/// Describes the Score calculation process for the Neuron of attention layer
/// (#CNeuronAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void AttentionScore(__global float *querys, ///<[in] Matrix of Querys
                             __global float *keys,   ///<[in] Matrix of Keys
                             __global float *score,  ///<[out] Matrix of Scores
                             int dimension,          ///< Dimension of Key
                             int mask ///< 1 - calc only previous units, 0 - calc all
                            )
  {
   int q = get_global_id(0);
   int shift_q = q * dimension;
   int units = get_global_size(0);
   int shift_s = q * units;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0;
   for(int k = 0; k < units; k++)
     {
      if(mask > 0 && k > q)
        {
         score[shift_s + k] = 0;
         continue;
        }
      float result = 0;
      int shift_k = k * dimension;
      for(int i = 0; i < dimension; i++)
         result += (querys[shift_q + i] * keys[shift_k + i]);
      result = exp(result / koef);
      if(isnan(result))
         result = 0;
      score[shift_s + k] = result;
      sum += result;
     }
   for(int k = 0; (k < units && sum > 0); k++)
      score[shift_s + k] /= sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Attention Neuron Out calculation kernel
/// Describes the Attention out calculation process for the Neuron of attention
/// layer (#CNeuronAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void AttentionOut(__global float *scores, ///<[in] Matrix of Scores
                           __global float *values, ///<[in] Matrix of Values
                           __global float *inputs, ///<[in] Inputs tensor
                           __global float *out     ///<[out] Output tensor
                          )
  {
   int units = get_global_size(0);
   int u = get_global_id(0);
   int d = get_global_id(1);
   int dimension = get_global_size(1);
   int shift = u * dimension + d;
   float result = 0;
   for(int i = 0; i < units; i++)
      result += scores[u * units + i] * values[i * dimension + d];
   out[shift] = (isnan(result) ? 0 : result) + inputs[shift];
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Kernel for calculation Sum of 2 matrixs with
/// multiplyer.
/// Describes the calculation Sum of 2 matrixs.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8765#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void SumMatrix(__global float *matrix1,    ///<[in] First matrix
                        __global float *matrix2,    ///<[in] Second matrix
                        __global float *matrix_out, ///<[out] Output matrix
                        int dimension,              ///< Dimension of matrix
                        float multiplyer,           ///< Multiplyer for output
                        int shift_in1,              ///< Shift for input 1
                        int shift_in2,              ///< Shift for input 2
                        int shift_out               ///< Shift for output
                       )
  {
   const int i = get_global_id(0);
   const int step = get_global_size(0);
   ;
   for(int k = 0; k < dimension; k++)
     {
      int index = i * dimension + k;
      matrix_out[i * shift_out + index] =
         (matrix1[i * shift_in1 + index] + matrix2[i * shift_in2 + index]) * multiplyer;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Kernel for calculation Sum of 4 matrixs with
/// multiplyer.
/// Describes the calculation Sum of 4 matrixs.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8909#para53">the link.</A>
//+------------------------------------------------------------------+
__kernel void Sum5Matrix(__global float *matrix1,    ///<[in] First matrix
                         __global float *matrix2,    ///<[in] Second matrix
                         __global float *matrix3,    ///<[in] Third matrix
                         __global float *matrix4,    ///<[in] Fourth matrix
                         __global float *matrix5,    ///<[in] Fifth matrix
                         __global float *matrix_out, ///<[out] Output matrix
                         int dimension,              ///< Dimension of matrix
                         float multiplyer            ///< Multiplyer for output
                        )
  {
   const int i = get_global_id(0) * dimension;
   for(int k = 0; k < dimension; k++)
      matrix_out[i + k] = (matrix1[i + k] + matrix2[i + k] + matrix3[i + k] +
                           matrix4[i + k] + matrix5[i + k]) *
                          multiplyer;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr Attention layer's neuron Gradients Calculation
/// kernel
/// Describes the gradients calculation process for the Neuron of attention
/// layer (#CNeuronAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8765#para44">the link.</A>
/// @param[in] querys Matrix of Querys
/// @param[out] querys_g Matrix of Querys' Gradients
/// @param[in] keys Matrix of Keys
/// @param[out] keys_g Matrix of Keys' Gradients
/// @param[in] values Matrix of Values
/// @param[out] values_g Matrix of Values' Gradients
/// @param[in] scores Matrix of Scores
/// @param[in] gradient Matrix of Gradients from previous iteration
//+------------------------------------------------------------------+
__kernel void AttentionInsideGradients(__global float *querys, __global float *querys_g,
                                       __global float *keys, __global float *keys_g,
                                       __global float *values, __global float *values_g,
                                       __global float *scores, __global float *gradient)
  {
   int u = get_global_id(0);
   int d = get_global_id(1);
   int units = get_global_size(0);
   int dimension = get_global_size(1);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float vg = 0;
   float qg = 0;
   float kg = 0;
   for(int iu = 0; iu < units; iu++)
     {
      float g = gradient[iu * dimension + d];
      float sc = scores[iu * units + u];
      vg += sc * g;
      //---
      float sqg = 0;
      float skg = 0;
      for(int id = 0; id < dimension; id++)
        {
         sqg += values[iu * dimension + id] * gradient[u * dimension + id];
         skg += values[u * dimension + id] * gradient[iu * dimension + id];
        }
      qg += (scores[u * units + iu] == 0 || scores[u * units + iu] == 1
             ? 0.0001f
             : scores[u * units + iu] * (1 - scores[u * units + iu])) *
            sqg * keys[iu * dimension + d] / koef;
      //---
      kg += (scores[iu * units + u] == 0 || scores[iu * units + u] == 1
             ? 0.0001f
             : scores[iu * units + u] * (1 - scores[iu * units + u])) *
            skg * querys[iu * dimension + d] / koef;
     }
   int shift = u * dimension + d;
   values_g[shift] = clamp((isnan(vg) ? 0.0f : vg), -1.0f, 1.0f);
   querys_g[shift] = clamp((isnan(qg) ? 0.0f : qg), -1.0f, 1.0f);
   keys_g[shift] = clamp((isnan(kg) ? 0.0f : kg), -1.0f, 1.0f);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of matrix normalization process
/// Describes the process of matrix normalization.
///\details Detailed description on <A
/// HREF="https://arxiv.org/abs/1607.06450">the link.</A>
/// @param[in,out] buffer In/Out Matrix
/// @param[in] dimension Dimension of matrix
//+------------------------------------------------------------------+
__kernel void Normalize(__global float *buffer, int dimension)
  {
   int n = get_global_id(0);
   int shift = n * dimension;
   if(dimension <= 0)
      return;
//---
   float mean = 0;
   for(int i = 0; i < dimension; i++)
     {
      float val = buffer[shift + i];
      if(isnan(val) || isinf(val))
         buffer[shift + i] = 0;
      else
         mean += val / dimension;
     }
   float variance = 0;
   for(int i = 0; i < dimension; i++)
      variance += pow(buffer[shift + i] - mean, 2) / dimension;
   variance = sqrt((isnan(variance) || isinf(variance) ? 0 : variance));
   for(int i = 0; i < dimension; i++)
      buffer[shift + i] =
         (buffer[shift + i] - mean) / (variance > 0 ? variance : 1.0f);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of weights matrix normalization process
/// Describes the process of weights matrix normalization.
///\details Detailed description on <A
/// HREF="https://arxiv.org/abs/1607.06450">the link.</A>
/// @param[in,out] buffer In/Out Matrix
/// @param[in] dimension Dimension of matrix
//+------------------------------------------------------------------+
__kernel void NormalizeWeights(__global float *buffer, int dimension)
  {
   int n = get_global_id(0);
   int shift = n * dimension;
   float sum = 0;
   float k = 1;
   do
     {
      for(int i = 0; (i < dimension && !isnan(sum) && !isinf(sum)); i++)
         sum = pow(buffer[shift + i] / k, 2) / dimension;
      if(isnan(sum))
         k *= 10;
     }
   while(isnan(sum) || isinf(sum));
   sum = sqrt(sum);
   if(k * sum > 1)
      for(int i = 0; i < dimension; i++)
         buffer[shift + i] /= k * sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff
/// Describes the process of concatenate 4 matrices.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8909#para52">the link.</A>
/// @param[in] input1, input2, input3, input4 Input buffers
/// @param[in] window1, window2, window3, window4 Windows for every buffer
/// @param[out] output Output buffer
//+------------------------------------------------------------------+
__kernel void ConcatenateBuffers(__global float *input1, int window1,
                                 __global float *input2, int window2,
                                 __global float *input3, int window3,
                                 __global float *input4, int window4,
                                 __global float *output)
  {
   int n = get_global_id(0);
   int shift = n * (window1 + window2 + window3 + window4);
   int shift_in = n * window1;
   for(int i = 0; i < window1; i++)
      output[shift + i] = input1[shift_in + i];
//---
   shift += window1;
   shift_in = n * window2;
   for(int i = 0; i < window2; i++)
      output[shift + i] = input2[shift_in + i];
//---
   shift += window2;
   shift_in = n * window3;
   for(int i = 0; i < window3; i++)
      output[shift + i] = input3[shift_in + i];
//---
   shift += window3;
   shift_in = n * window4;
   for(int i = 0; i < window4; i++)
      output[shift + i] = input4[shift_in + i];
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr
/// Describes the process of deconcatenate matrix.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/8909#para53">the link.</A>
/// @param[in] output1, output2, output3, output4 Output buffers
/// @param[in] window1, window2, window3, window4 Windows for every buffer
/// @param[out] inputs Input buffer
//+------------------------------------------------------------------+
__kernel void DeconcatenateBuffers(__global float *output1, int window1,
                                   __global float *output2, int window2,
                                   __global float *output3, int window3,
                                   __global float *output4, int window4,
                                   __global float *inputs)
  {
   int n = get_global_id(0);
//--- Head 1
   int shift = n * (window1 + window2 + window3 + window4);
   int shift_out = n * window1;
   for(int i = 0; i < window1; i++)
      output1[shift_out + i] = inputs[shift + i];
//--- Head 2
   shift += window1;
   shift_out = n * window2;
   for(int i = 0; i < window2; i++)
      output2[shift_out + i] = inputs[shift + i];
//--- Head 3
   shift += window2;
   if(window3 > 0)
     {
      shift_out = n * window3;
      for(int i = 0; i < window3; i++)
         output3[shift_out + i] = inputs[shift + i];
     }
//--- Head 4
   shift += window3;
   if(window4 > 0)
     {
      shift_out = n * window4;
      for(int i = 0; i < window4; i++)
         output4[shift_out + i] = inputs[shift + i];
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Multi-Heads Attention Neuron Score calculation
/// kernel
/// Describes the Score calculation process for the Neuron of multi-heads
/// attention layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9025#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void MHAttentionScore(__global float *qkv,   ///<[in] Matrix of Querys, Keys, Values
                               __global float *score, ///<[out] Matrix of Scores
                               int dimension,         ///< Dimension of Key
                               int mask ///< 1 - calc only previous units, 0 - calc all
                              )
  {
   int q = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
//---
   int shift_q = dimension * (h + 3 * q * heads);
   int shift_s = units * (h + q * heads);
//---
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0;
   for(int k = 0; k < units; k++)
     {
      if(mask > 0 && k > q)
        {
         score[shift_s + k] = 0;
         continue;
        }
      float result = 0;
      int shift_k = dimension * (h + heads * (3 * k + 1));
      for(int i = 0; i < dimension; i++)
        {
         if((dimension - i) > 4)
           {
            result += dot((float4)(qkv[shift_q + i], qkv[shift_q + i + 1],
                                   qkv[shift_q + i + 2], qkv[shift_q + i + 3]),
                          (float4)(qkv[shift_k + i], qkv[shift_k + i + 1],
                                   qkv[shift_k + i + 2], qkv[shift_k + i + 3]));
            i += 3;
           }
         else
            result += (qkv[shift_q + i] * qkv[shift_k + i]);
        }
      result = exp(clamp(result / koef, -30.0f, 30.0f));
      if(isnan(result))
         result = 0;
      score[shift_s + k] = result;
      sum += result;
     }
   for(int k = 0; (k < units && sum > 1); k++)
      score[shift_s + k] /= sum;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_ff Multi-heads Attention Neuron Out calculation kernel
/// Describes the Multi-heads Attention out calculation process for the Neuron
/// of multi-heads attention layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9025#para42">the link.</A>
//+------------------------------------------------------------------+
__kernel void MHAttentionOut(__global float *scores, ///<[in] Matrix of Scores
                             __global float *qkv,    ///<[in] Matrix of Values
                             __global float *out,    ///<[out] Output tensor
                             int dimension           ///< Dimension of Value
                            )
  {
   int u = get_global_id(0);
   int units = get_global_size(0);
   int h = get_global_id(1);
   int heads = get_global_size(1);
//---
   int shift_s = units * (h + heads * u);
   int shift_out = dimension * (h + heads * u);
   int layer = 3 * dimension * heads;
//---
   for(int d = 0; d < dimension; d++)
     {
      float result = 0;
      for(int v = 0; v < units; v++)
        {
         int shift_v = dimension * (h + heads * (3 * v + 2)) + d;
         result += scores[shift_s + v] * qkv[shift_v];
        }
      out[shift_out + d] = result;
     }
  }
//+------------------------------------------------------------------+
///\ingroup neuron_atten_gr Attention layer's neuron Gradients Calculation
/// kernel
/// Describes the gradients calculation process for the Neuron of attention
/// layer (#CNeuronMLMHAttentionOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9025#para33">the link.</A>
/// @param[in] qkv Matrix of Querys, Keys and Values
/// @param[out] qkv_g Matrix of Querys', Keys' and Values' Gradients
/// @param[in] scores Matrix of Scores
/// @param[in] scores_g Matrix of Scores' Gradients
/// @param[in] gradient Matrix of Gradients from previous iteration
/// @param[in] dimension Dimension of Key vector
//+------------------------------------------------------------------+
__kernel void MHAttentionInsideGradients(__global float *qkv, __global float *qkv_g,
      __global float *scores,
      __global float *gradient)
  {
   size_t u = get_global_id(0);
   size_t h = get_global_id(1);
   size_t d = get_global_id(2);
   size_t units = get_global_size(0);
   size_t heads = get_global_size(1);
   size_t dimension = get_global_size(2);
//---
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- init
   const int shift_q = dimension * (heads * 3 * u + h);
   const int shift_k = dimension * (heads * (3 * u + 1) + h);
   const int shift_v = dimension * (heads * (3 * u + 2) + h);
   const int shift_g = dimension * (heads * u + h);
   int shift_score = h * units;
   int step_score = units * heads;
//--- Calculating Value's gradients
   float sum = 0;
   for(int i = 0; i < units; i++)
      sum += gradient[(h + i * heads) * dimension + d] * scores[shift_score + u + i * step_score];
   qkv_g[shift_v + d] = sum;
//--- Calculating Query's gradients
   shift_score = h * units + u * step_score;
   float grad = 0;
   float grad_out = gradient[shift_g + d];
   for(int k = 0; k < units; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_score + k];
      for(int v = 0; v < units; v++)
         sc_g += scores[shift_score + v] * qkv[dimension * (heads * (3 * v + 2) + h)] *
                 grad_out * ((k == v) - sc);
      grad += sc_g / koef * qkv[dimension * (heads * (3 * k + 1) + h) + d];
     }
   qkv_g[shift_q + d] = grad;
//--- Calculating Key's gradients
   grad = 0;
   for(int q = 0; q < units; q++)
     {
      shift_score = h * units + q * step_score;
      float sc_g = 0;
      float sc = scores[shift_score + u];
      float grad_out = gradient[dimension * (heads * q + h) + d];
      for(int v = 0; v < units; v++)
         sc_g += scores[shift_score + v] * qkv[dimension * (heads * (3 * v + 2) + h)] *
                 grad_out * ((u == v) - sc);
      grad += sc_g / koef * qkv[dimension * (heads * 3 * q + h) + d];
     }
   qkv_g[shift_k + d] = grad;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_dropout Kernel for Dropout.
/// Describes the dropout method.
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9112#para32">the link.</A>
//+------------------------------------------------------------------+
__kernel void Dropout(__global const float *inputs, ///<[in] Input matrix
                      __global const float *map,    ///<[in] Dropout map matrix
                      __global float *out,    ///<[out] Output matrix
                      const int dimension           ///< Dimension of matrix
                     )
  {
   const int i = get_global_id(0) * 4;
   if(i + 3 < dimension)
     {
      float4 k =
         (float4)(inputs[i], inputs[i + 1], inputs[i + 2], inputs[i + 3]) *
         (float4)(map[i], map[i + 1], map[i + 2], map[i + 3]);
      out[i] = k.s0;
      out[i + 1] = k.s1;
      out[i + 2] = k.s2;
      out[i + 3] = k.s3;
     }
   else
      for(int k = i; k < min(dimension, i + 4); k++)
         out[i + k] = (inputs[i + k] * map[i + k]);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_norm Kernels of Batch normalization process
/// Describes the process of Batch normalization. (#CNeuronBatchNormOCL)
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9207#para42">the link.</A>
/// @param[in] inputs Input data tenzor
/// @param[in,out] options Tenzor of variables
/// @param[out] output Tenzor of output data
/// @param[in] batch Batch size
/// @param[in] optimization Optimization type
/// @param[in] activation Activation type
//+------------------------------------------------------------------+
__kernel void BatchFeedForward(__global float *inputs, __global float *options,
                               __global float *output, int batch,
                               int optimization, int activation)
  {
   if(batch <= 1)
      return;
   int n = get_global_id(0);
   int shift = n * (optimization == 0 ? 7 : 9);
//---
   for(int i = 0; i < (optimization == 0 ? 7 : 9); i++)
     {
      float opt = options[shift + i];
      if(isnan(opt) || isinf(opt))
         options[shift + i] = 0;
     }
//---
   float inp = inputs[n];
   float mean = (batch > 1 ? (options[shift] * ((float)batch - 1.0f) + inp) / ((float)batch) : inp);
   float delt = inp - mean;
   float variance = options[shift + 1] * ((float)batch - 1.0f) + pow(delt, 2);
   if(batch > 0)
      variance /= (float)batch;
   float nx = (variance > 0 ? delt / sqrt(variance) : 0);
//---
   float gamma = options[shift + 3];
   if(gamma == 0 || isinf(gamma) || isnan(gamma))
     {
      options[shift + 3] = 1;
      gamma = 1;
     }
   float betta = options[shift + 4];
   if(isinf(betta) || isnan(betta))
     {
      options[shift + 4] = 0;
      betta = 0;
     }
//---
   options[shift] = mean;
   options[shift + 1] = variance;
   options[shift + 2] = nx;
   output[n] = Activation(gamma * nx + betta, activation);;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_gr
/// Kernel of the Batch neuron to transfer gradient to previous layer
/// (#CNeuronBatchNormOCL)
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientBatch(__global float *options,      ///<[in] Options matrix m*(7 or 9), where m - Number of neurons in previous layer
                                      __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                      __global float *matrix_i,  ///<[in] Tensor of previous layer output
                                      __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
                                      int activation,            ///< Activation type (#ENUM_ACTIVATION)
                                      int batch,                 ///< Batch size
                                      int optimization           ///< Optimization type
                                     )
  {
   if(batch <= 1)
      return;
//---
   int n = get_global_id(0);
   int shift = n * (optimization == 0 ? 7 : 9);
//---
   float variance = options[shift + 1];
//---
   float inp = matrix_i[n];
   float gnx = matrix_g[n] * options[shift + 3];
   float temp = (variance > 0 ? 1.0f / sqrt(variance) : 0);
   float gmu = (-temp) * gnx;
   float gvar =
      (variance > 0
       ? (options[shift] * inp) / (2 * pow(variance, 3.0f / 2.0f)) * gnx
       : 0);
   float gx = temp * gnx + gmu / batch +
              gvar * 2 * inp / batch * pow((float)(batch - 1) / batch, 2.0f);
//---
   matrix_ig[n] = Deactivation(gx, inp, activation);;
  }
//+------------------------------------------------------------------+
///\ingroup neuron_opt Batch normalization Neuron SGD optimization Updating
/// options kernel
/// Describes the process of SGD optimization options for the Batch
/// normalization Neuron (#CNeuronBatchNormOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateBatchOptionsMomentum(__global float *options,     ///<[in,out] Options matrix m*7, where m - Number of neurons in previous layer
      __global float *matrix_g, ///<[in] Tensor of gradients at current layer
      float learning_rates,     ///< Learning rates
      float momentum            ///< Momentum multiplier
                                        )
  {
   const int n = get_global_id(0);
   int inputs = get_global_size(0);
   const int shift = n * 7;
   float grad = matrix_g[n];
//---
   float2 delta = learning_rates * grad * (float2)(options[shift + 2], 1) +
                  momentum * (float2)(options[shift + 5], options[shift + 6]);
   if(isnan(delta.s0) || isinf(delta.s0))
      delta.s0 = 0;
   if(isnan(delta.s1) || isinf(delta.s1))
      delta.s1 = 0;
   options[shift + 5] = delta.s0;
   float value = options[shift + 3];
   options[shift + 3] += delta.s0 - learning_rates * (l1 * sign(value) +
                         l2 * value / inputs);
//---
   options[shift + 6] = delta.s1;
   value = options[shift + 4];
   options[shift + 4] += delta.s1 - learning_rates * (l1 * sign(value) +
                         l2 * value / inputs);
  }
//+------------------------------------------------------------------+
///\ingroup neuron_opt Batch normalization Neuron Adam optimization Updating
/// options kernel
/// Describes the process of Adam optimization options for the Batch
/// normalization  Neuron (#CNeuronBatchNormOCL).
///\details Detailed description on <A
/// HREF="https://www.mql5.com/ru/articles/9207#para43">the link.</A>
//+------------------------------------------------------------------+
__kernel void UpdateBatchOptionsAdam(__global float *options,     ///<[in,out] Options matrix m*9, where m - Number of neurons in previous layer
                                     __global float *matrix_g, ///<[in] Tensor of gradients at current layer
                                     const float l,            ///< Learning rates
                                     const float b1,           ///< First momentum multiplier
                                     const float b2            ///< Second momentum multiplier
                                    )
  {
   const int n = get_global_id(0);
   int inputs = get_global_size(0);
   const int shift = n * 9;
   float grad = matrix_g[n];
//---
   float gamma = options[shift + 3];
   if(isnan(gamma) || isinf(gamma))
     {
      options[shift + 3] = 1;
      gamma = 1;
     }
   float betta = options[shift + 4];
   if(isinf(betta) || isnan(betta))
     {
      options[shift + 4] = 0;
      betta = 0;
     }
//---
   float gamma_m1 = options[shift + 5];
   if(isnan(gamma_m1) || isinf(gamma_m1))
     {
      options[shift + 5] = 0;
      gamma_m1 = 0;
     }
   float betta_m1 = options[shift + 6];
   if(isinf(betta_m1) || isnan(betta_m1))
     {
      options[shift + 6] = 0;
      betta_m1 = 0;
     }
//---
   float gamma_m2 = options[shift + 7];
   if(isnan(gamma_m2) || isinf(gamma_m2))
     {
      options[shift + 7] = 0;
      gamma_m2 = 0;
     }
   float betta_m2 = options[shift + 8];
   if(isinf(betta_m2) || isnan(betta_m2))
     {
      options[shift + 8] = 0;
      betta_m1 = 0;
     }
//---
   float2 mt = b1 * (float2)(gamma_m1, betta_m1) +
               (1 - b1) * (float2)(grad * options[shift + 2],
                                   grad);
   float2 vt = b2 * (float2)(gamma_m2, betta_m2) +
               (1 - b2) * pow((float2)(grad * options[shift + 2], grad), 2);
   if(isinf(vt.s0) || isnan(vt.s0))
      vt.s0 = 0;
   if(isinf(vt.s1) || isnan(vt.s1))
      vt.s1 = 0;
   float2 delta = l * mt / sqrt(vt);
   if(isnan(delta.s0) || isinf(delta.s0))
      delta.s0 = 0;
   if(isnan(delta.s1) || isinf(delta.s1))
      delta.s0 = 0;
   float2 weight = delta -
                   (l1 * sign((float2)(gamma, betta)) +
                    l2 * (float2)(gamma, betta) / inputs);
//---
   if(!(isnan(gamma + weight.s0) || isinf(gamma + weight.s0)))
      options[shift + 3] = gamma + weight.s0;
   if(!(isnan(betta + weight.s1) || isinf(betta + weight.s1)))
      options[shift + 4] = betta + weight.s1;
   if(!(isnan(mt.s0) || isinf(mt.s0)))
      options[shift + 5] = mt.s0;
   if(!(isnan(mt.s1) || isinf(mt.s1)))
      options[shift + 6] = mt.s1;
   if(!(isnan(vt.s0) || isinf(vt.s0)))
      options[shift + 7] = vt.s0;
   if(!(isnan(vt.s1) || isinf(vt.s1)))
      options[shift + 8] = vt.s1;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void VAE_FeedForward(__global float *inputs, __global float *random,
                              __global float *outputs)
  {
   uint i = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   outputs[i] = inputs[i] + exp(0.5f * inputs[i + total]) * random[i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void VAE_CalcHiddenGradient(__global float *inputs,
                                     __global float *inp_grad,
                                     __global float *random,
                                     __global float *gradient,
                                     const float kld_mult)
  {
   uint i = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   float log_var = inputs[i + total];
   float kld =
      kld_mult * 0.5f * (log_var - exp(log_var) - pow(inputs[i], 2.0f) + 1);
   float grad = gradient[i];
   inp_grad[i] = grad / exp(0.5f * log_var) + kld * inputs[i];
   inp_grad[i + total] = 0.5f * (grad * random[i] - kld * (1 - exp(log_var)));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void LSTM_FeedForward(__global const float *inputs, int inputs_size,
                               __global const float *weights,
                               __global float *concatenated,
                               __global float *memory, __global float *output)
  {
   uint id = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   uint id2 = (uint)get_local_id(1);
   uint idv = (uint)get_global_id(2);
   uint total_v = (uint)get_global_size(2);
//---
   __local float Temp[4];
//---
   float sum = 0;
   uint shift_in = idv * inputs_size;
   uint shift_out = idv * total;
   uint shift = (inputs_size + total + 1) * (id2 + id);
//---
   for(uint i = 0; i < total; i += 4)
     {
      if(total - i > 4)
         sum +=
            dot((float4)(output[shift_out + i], output[shift_out + i + 1], output[shift_out + i + 2], output[shift_out + i + 3]),
                (float4)(weights[shift + i], weights[shift + i + 1],
                         weights[shift + i + 2], weights[shift + i + 3]));
      else
         for(uint k = i; k < total; k++)
            sum += output[shift_out + k] * weights[shift + k];
     }
//---
   shift += total;
   for(uint i = 0; i < inputs_size; i += 4)
     {
      if(total - i > 4)
         sum +=
            dot((float4)(inputs[shift_in + i], inputs[shift_in + i + 1], inputs[shift_in + i + 2], inputs[shift_in + i + 3]),
                (float4)(weights[shift + i], weights[shift + i + 1],
                         weights[shift + i + 2], weights[shift + i + 3]));
      else
         for(uint k = i; k < total; k++)
            sum += inputs[shift_in + k] * weights[shift + k];
     }
   sum += weights[shift + inputs_size];
   if(isnan(sum) || isinf(sum))
      sum = 0;
   if(id2 < 3)
      sum = Activation(sum, 1);
   else
      sum = Activation(sum, 0);
   Temp[id2] = sum;
   concatenated[4 * shift_out + id2 * total + id] = sum;
//---
   barrier(CLK_LOCAL_MEM_FENCE);
   if(id2 == 0)
     {
      float mem = memory[shift_out + id + total_v * total] = memory[shift_out + id];
      float fg = Temp[0];
      float ig = Temp[1];
      float og = Temp[2];
      float nc = Temp[3];
      //---
      memory[shift_out + id] = mem = mem * fg + ig * nc;
      output[shift_out + id] = og * Activation(mem, 0);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void LSTM_ConcatenatedGradient(__global float *gradient,
                                        __global float *concatenated_gradient,
                                        __global float *memory,
                                        __global float *concatenated)
  {
   uint id = (uint)get_global_id(0);
   uint total = (uint)get_global_size(0);
   uint idv = (uint)get_global_id(1);
   uint total_v = (uint)get_global_size(1);
//---
   uint shift_out = idv * total;
   float t = tanh(memory[shift_out + id]);
//---
   concatenated_gradient[4 * shift_out + id + 2 * total] = gradient[shift_out + id] * t; // output gate
//---
   float memory_gradient = gradient[shift_out + id] * concatenated[4 * shift_out + id + 2 * total];
   memory_gradient *= 1 - pow(t, 2.0f);
//---
   concatenated_gradient[4 * shift_out + id + 3 * total] =
      memory_gradient * concatenated[4 * shift_out + id + total]; // new content
//---
   concatenated_gradient[4 * shift_out + id + total] =
      memory_gradient * concatenated[4 * shift_out + id + 3 * total]; // input gate
//---
   concatenated_gradient[4 * shift_out + id] =
      memory_gradient * memory[shift_out + id + total_v * total]; // forgat gate
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void LSTM_HiddenGradient(__global float *concatenated_gradient, __global float *inputs_gradient,
                                  __global float *weights_gradient, __global float *hidden_state,
                                  __global float *inputs, __global float *weights, __global float *output,
                                  const int hidden_size, const int inputs_size)
  {
   uint id = get_global_id(0);
   uint total = get_global_size(0);
   uint idv = (uint)get_global_id(1);
   uint total_v = (uint)get_global_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   uint ls = min(total_v, (uint)LOCAL_ARRAY_SIZE);
//---
   uint shift_in = idv * inputs_size;
   uint shift_out = idv * total;
   uint weights_step = hidden_size + inputs_size + 1;
//---
   for(int i = id; i < (hidden_size + inputs_size); i += total)
     {
      float inp = 0;
      if(i < hidden_size)
        {
         inp = hidden_state[shift_out + i];
         hidden_state[shift_out + i] = output[shift_out + i];
        }
      else
        {
         inp = inputs[shift_in + i - hidden_size];
         float grad = 0;
         for(uint g = 0; g < 3 * hidden_size; g++)
           {
            float temp = concatenated_gradient[4 * shift_out + g];
            grad += temp * (1 - temp) * weights[i + g * weights_step];
           }
         for(uint g = 3 * hidden_size; g < 4 * hidden_size; g++)
           {
            float temp = concatenated_gradient[4 * shift_out + g];
            grad += temp * (1 - pow(temp, 2.0f)) * weights[i + g * weights_step];
           }
         inputs_gradient[shift_in + i - hidden_size] = grad;
        }
      //---
      for(uint g = 0; g < 3 * hidden_size; g++)
        {
         float temp = concatenated_gradient[4 * shift_out + g];
         if(idv < ls)
            Temp[idv % ls] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
         for(uint v = 0; v < total_v; v += ls)
           {
            if(idv >= v && idv < v + ls)
               Temp[idv % ls] += temp * (1 - temp) * inp;
            barrier(CLK_LOCAL_MEM_FENCE);
           }
         if(idv == 0)
           {
            temp = Temp[0];
            for(int v = 1; v < ls; v++)
               temp += Temp[v];
            weights_gradient[i + g * weights_step] = temp;
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      for(uint g = 3 * hidden_size; g < 4 * hidden_size; g++)
        {
         float temp = concatenated_gradient[4 * shift_out + g];
         if(idv < ls)
            Temp[idv % ls] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
         for(uint v = 0; v < total_v; v += ls)
           {
            if(idv >= v && idv < v + ls)
               Temp[idv % ls] += temp * (1 - pow(temp, 2.0f)) * inp;
            barrier(CLK_LOCAL_MEM_FENCE);
           }
         if(idv == 0)
           {
            temp = Temp[0];
            for(int v = 1; v < ls; v++)
               temp += Temp[v];
            weights_gradient[i + g * weights_step] = temp;
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
     }
//---
   for(int i = id; i < 4 * hidden_size; i += total)
     {
      if(idv < ls)
         Temp[idv % ls] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
      float temp = concatenated_gradient[4 * shift_out + (i + 1) * hidden_size];
      if(i < 3 * hidden_size)
        {
         for(uint v = 0; v < total_v; v += ls)
           {
            if(idv >= v && idv < v + ls)
               Temp[idv % ls] += temp * (1 - temp);
            barrier(CLK_LOCAL_MEM_FENCE);
           }
        }
      else
        {
         for(uint v = 0; v < total_v; v += ls)
           {
            if(idv >= v && idv < v + ls)
               Temp[idv % ls] += 1 - pow(temp, 2.0f);
            barrier(CLK_LOCAL_MEM_FENCE);
           }
        }
      if(idv == 0)
        {
         temp = Temp[0];
         for(int v = 1; v < ls; v++)
            temp += Temp[v];
         weights_gradient[(i + 1) * weights_step] = temp;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
  }
//+------------------------------------------------------------------+
///\ingroup LSTM_opt  LSTM Adam Updating Weights Calculation kernel
/// Describes the process of Adam optimization weights for the Neuron LSTM
/// (#CNeuronLSTMOCL).
//+------------------------------------------------------------------+
__kernel void LSTM_UpdateWeightsAdam(__global float *weights,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                     ///< number of neurons in previous layer and n -
                                     ///< number of neurons in current layer
                                     __global float
                                     *weights_gradient,    ///<[in] Tensor of gradients at current layer
                                     __global float *matrix_m, ///<[in,out] Matrix of first momentum
                                     __global float *matrix_v, ///<[in,out] Matrix of seconfd momentum
                                     const float l,            ///< Learning rates
                                     const float b1,           ///< First momentum multiplier
                                     const float b2            ///< Second momentum multiplier
                                    )
  {
   const uint id = get_global_id(0);
   const uint total = get_global_size(0);
   const uint id1 = get_global_id(1);
   const uint wi = id1 * total + id;
   float g = weights_gradient[wi];
   float mt = b1 * matrix_m[wi] + (1 - b1) * g;
   float vt = b2 * matrix_v[wi] + (1 - b2) * pow(g, 2);
   float delta = l * (mt / (sqrt(vt) + 1.0e-37f) -
                      (l1 * sign(weights[wi]) + l2 * weights[wi] / total));
   weights[wi] = clamp(weights[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[wi] = mt;
   matrix_v[wi] = vt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftMax_FeedForward(__global float *inputs,
                                  __global float *outputs, const int total)
  {
   const uint i = (uint)get_global_id(0);
   const uint l = (uint)get_local_id(0);
   const uint h = (uint)get_global_id(1);
   const uint ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
   uint shift_head = h * total;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   uint count = 0;
   if(l < ls)
      do
        {
         uint shift = shift_head + count * ls + l;
         if(shift < ((h + 1) * total))
            temp[l] = (count > 0 ? fmax(fabs(inputs[shift]), temp[l])
                       : fabs(inputs[shift]));
         count++;
        }
      while((count * ls + l) < total);
   barrier(CLK_LOCAL_MEM_FENCE);
   float max_value = temp[0];
   for(int i = 1; i < ls; i++)
      max_value = fmax(max_value, temp[i]);
//---
   count = 0;
   if(l < ls)
      do
        {
         uint shift = shift_head + count * ls + l;
         temp[l] =
            (count > 0 ? temp[l] : 0) +
            (shift < ((h + 1) * total) ? exp(inputs[shift] / max_value) : 0);
         count++;
        }
      while((count * ls + l) < total);
   barrier(CLK_LOCAL_MEM_FENCE);
   count = min(ls, (uint)total);
   do
     {
      count = (count + 1) / 2;
      if(l < ls)
         temp[l] += (l < count && (l + count) < total ? temp[l + count] : 0);
      if(l + count < ls)
         temp[l + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   float sum = temp[0];
   if(sum != 0)
     {
      count = 0;
      while((count * ls + l) < total)
        {
         uint shift = shift_head + count * ls + l;
         if(shift < ((h + 1) * total))
            outputs[shift] = exp(inputs[shift] / max_value) / (sum + 1e-37f);
         count++;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftMax_HiddenGradient(__global float *outputs,
                                     __global float *output_gr,
                                     __global float *input_gr)
  {
   size_t i = get_global_id(0);
   size_t outputs_total = get_global_size(0);
   size_t h = get_global_id(1);
   uint shift = h * outputs_total;
   float output = outputs[shift + i];
   float result = 0;
   for(int j = 0; j < outputs_total; j++)
      result +=
         outputs[shift + j] * output_gr[shift + j] * ((float)(i == j) - output);
   input_gr[shift + i] = result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftMax_OutputGradient(__global float *outputs,
                                     __global float *targets,
                                     __global float *output_gr)
  {
   size_t i = get_global_id(0);
   output_gr[i] = (outputs[i] == 0 ? 0 : targets[i] / outputs[i]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_Cosine(__global float *softmax, __global float *output)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float result = 0;
   for(int it = 0; it < i; it++)
      result += softmax[shift + it];
   result += softmax[shift + i] / 2.0f;
   output[shift + i] = cos(i * M_PI_F * result);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_Output(__global float *quantiles, __global float *delta_taus,
                         __global float *output, int total)
  {
   size_t action = get_global_id(0);
   int shift = action * total;
//---
   float result = 0;
   for(int i = 0; i < total; i++)
      result += quantiles[shift + i] * delta_taus[shift + i];
   output[action] = result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_OutputGradient(__global float *quantiles,
                                 __global float *delta_taus,
                                 __global float *output_gr,
                                 __global float *quantiles_gr,
                                 __global float *taus_gr)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float gradient = output_gr[action];
   quantiles_gr[shift + i] = gradient * delta_taus[shift + i];
   taus_gr[shift + i] = gradient * quantiles[shift + i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_QuantileGradient(__global float *state_embeding,
                                   __global float *taus_embeding,
                                   __global float *quantiles_gr,
                                   __global float *state_gr,
                                   __global float *taus_gr)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float gradient = quantiles_gr[shift + i];
   state_gr[shift + i] = gradient * taus_embeding[shift + i];
   taus_gr[shift + i] = gradient * state_embeding[shift + i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FQF_CosineGradient(__global float *softmax,
                                 __global float *output_gr,
                                 __global float *softmax_gr)
  {
   size_t i = get_global_id(0);
   size_t total = get_global_size(0);
   size_t action = get_global_id(1);
   int shift = action * total;
//---
   float cumul = 0;
   for(int it = 0; it < i; it++)
      cumul += softmax[shift + it];
   float result = -M_PI_F * i *
                  sin(M_PI_F * i * (cumul + softmax[shift + i] / 2)) *
                  output_gr[shift + i];
   for(int it = i + 1; it < total; it++)
     {
      cumul += softmax[shift + it - 1];
      float temp = cumul + softmax[shift + it] / 2;
      result += -M_PI_F * it * sin(M_PI_F * it * temp) * output_gr[shift + it] *
                softmax[shift + it] / temp;
     }
   softmax_gr[shift + i] += result;
  }
//+------------------------------------------------------------------+
//| Sparse Attention                                                 |
//+------------------------------------------------------------------+
__kernel void MHSparseAttentionScore(__global float *qkv,      ///<[in] Matrix of Querys, Keys, Values
                                     __global float *score, ///<[out] Matrix of Scores
                                     int dimension,         ///< Dimension of Key
                                     float sparse           ///< less than 1.0 coefficient of sparse
                                    )
  {
   int q = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
//---
   int shift_q = dimension * (h + 3 * q * heads);
   int shift_s = units * (h + q * heads);
   int active_units = (int)max((float)(units * sparse), min((float)units, 3.0f));
//---
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   float sum = 0.0f;
   float min_s = 0.0f;
   float max_s = 0.0f;
   for(int k = 0; k < units; k++)
     {
      float result = 0;
      int shift_k = dimension * (h + heads * (3 * k + 1));
      for(int i = 0; i < dimension; i++)
        {
         if((dimension - i) > 4)
           {
            result += dot((float4)(qkv[shift_q + i], qkv[shift_q + i + 1],
                                   qkv[shift_q + i + 2], qkv[shift_q + i + 3]),
                          (float4)(qkv[shift_k + i], qkv[shift_k + i + 1],
                                   qkv[shift_k + i + 2], qkv[shift_k + i + 3]));
            i += 3;
           }
         else
            result += (qkv[shift_q + i] * qkv[shift_k + i]);
        }
      score[shift_s + k] = result;
      if(k == 0)
         min_s = max_s = result;
      else
        {
         max_s = max(max_s, result);
         min_s = min(min_s, result);
        }
     }
//---
   int count = units;
   while(count > active_units && min_s < max_s)
     {
      count = 0;
      float temp = max_s;
      for(int k = 0; k < units; k++)
        {
         float value = score[shift_s + k];
         if(value < min_s)
            continue;
         count++;
         if(value < temp && value > min_s)
            temp = value;
        }
      if(count > active_units)
         min_s = temp;
     }
//---
   if(max_s == 0.0f)
      max_s = 1.0f;
   for(int k = 0; k < units; k++)
     {
      float value = score[shift_s + k];
      if(value < min_s)
        {
         score[shift_s + k] = 0.0f;
         continue;
        }
      value = exp(value / max_s / koef);
      score[shift_s + k] = value;
      sum += value;
     }
//---
   for(int k = 0; (k < units && sum > 1); k++)
     {
      float temp = score[shift_s + k];
      if(temp == 0.0f)
         continue;
      score[shift_s + k] = temp / sum;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MHSparseAttentionOut(__global float *scores, ///<[in] Matrix of Scores
                                   __global float *qkv,    ///<[in] Matrix of Values
                                   __global float *out,    ///<[out] Output tensor
                                   int dimension           ///< Dimension of Value
                                  )
  {
   int u = get_global_id(0);
   int units = get_global_size(0);
   int h = get_global_id(1);
   int heads = get_global_size(1);
//---
   int shift_s = units * (h + heads * u);
   int shift_out = dimension * (h + heads * u);
//---
   for(int d = 0; d < dimension; d++)
     {
      float result = 0;
      for(int v = 0; v < units; v++)
        {
         float cur_score = scores[shift_s + v];
         if(cur_score == 0)
            continue;
         int shift_v = dimension * (h + heads * (3 * v + 2)) + d;
         result += cur_score * qkv[shift_v];
        }
      out[shift_out + d] = result;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForwardMultiModels(__global float *matrix_w,    ///<[in] Weights matrix (m+1)*n, where m - number of neurons in layer and n - number of outputs (neurons in next layer)
                                     __global float *matrix_i, ///<[in] Inputs tensor
                                     __global float *matrix_o, ///<[out] Output tensor
                                     int inputs,               ///< Number of inputs
                                     int activation            ///< Activation type (#ENUM_ACTIVATION)
                                    )
  {
   int i = get_global_id(0);
   int outputs = get_global_size(0);
   int m = get_global_id(1);
   int models = get_global_size(1);
//---
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs + 1) * (i + outputs * m);
   int shift_in = inputs * m;
   int shift_out = outputs * m;
   for(int k = 0; k <= inputs; k = k + 4)
     {
      switch(inputs - k)
        {
         case 0:
            inp = (float4)(1, 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 1:
            inp = (float4)(matrix_i[shift_in + k], 1, 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1], 1, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], 0);
            break;
         case 3:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1],
                           matrix_i[shift_in + k + 2], 1);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
         default:
            inp = (float4)(matrix_i[shift_in + k], matrix_i[shift_in + k + 1],
                           matrix_i[shift_in + k + 2], matrix_i[shift_in + k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
   if(isnan(sum))
      sum = 0;
//---
   matrix_o[shift_out + i] = Activation(sum, activation);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientMultiModels(
   __global float *matrix_w,  ///<[in] Weights matrix (m+1)*n, where m - number
   ///< of neurons in previous layer and n - number
   ///< of neurons in current layer
   __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
   __global float *matrix_o,  ///<[in] Previous layer Output tensor
   __global float *matrix_ig, ///<[out] Tensor of gradients at previous layer
   int outputs,               ///< Number of outputs
   int activation,            ///< Activation type (#ENUM_ACTIVATION),
   int model)
  {
   int i = get_global_id(0);
   int inputs = get_global_size(0);
   int m = get_global_id(1);
   int models = get_global_size(1);
//---
   int shift_in = inputs * m;
   if(model >= 0 && model != m)
     {
      matrix_ig[shift_in + i] = 0;
      return;
     }
//---
   int shift_out = outputs * m;
   int shift_w = (inputs + 1) * outputs * m;
   float sum = 0;
   float out = matrix_o[shift_in + i];
   float4 grad, weight;
   for(int k = 0; k < outputs; k += 4)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[shift_out + k], 0, 0, 0);
            break;
         case 2:
            grad =
               (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1], 0, 0);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1],
                            matrix_g[shift_out + k + 2], 0);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 1) * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[shift_out + k], matrix_g[shift_out + k + 1],
                            matrix_g[shift_out + k + 2], matrix_g[shift_out + k + 3]);
            weight = (float4)(matrix_w[shift_w + k * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 1) * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 2) * (inputs + 1) + i],
                              matrix_w[shift_w + (k + 3) * (inputs + 1) + i]);
            break;
        }
      sum += dot(grad, weight);
     }
//---
   matrix_ig[shift_in + i] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdamMultiModels(
   __global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
   ///< number of neurons in previous layer and n -
   ///< number of neurons in current layer
   __global const float
   *matrix_g, ///<[in] Tensor of gradients at current layer
   __global const float *matrix_i, ///<[in] Inputs tensor
   __global float *matrix_m,       ///<[in,out] Matrix of first momentum
   __global float *matrix_v,       ///<[in,out] Matrix of seconfd momentum
   const int inputs,               ///< Number of inputs
   const float l,                  ///< Learning rates
   const float b1,                 ///< First momentum multiplier
   const float b2,                 ///< Second momentum multiplier
   const int model)
  {
   const int outputs = get_global_size(0);
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   const int wi = (i + outputs * model) * (inputs + 1) + j * 4;
   float4 m, v, weight, inp;
   int shift_in = j * 4 + inputs * model;
   if((inputs + 1 - j * 4) < 0)
      return;
   switch(inputs + 1 - j * 4)
     {
      case 0:
         inp = (float4)(1, 0, 0, 0);
         weight = (float4)(matrix_w[wi], 0, 0, 0);
         m = (float4)(matrix_m[wi], 0, 0, 0);
         v = (float4)(matrix_v[wi], 0, 0, 0);
         break;
      case 1:
         inp = (float4)(matrix_i[shift_in], 1, 0, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], 0, 0);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], 0, 0);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], 0, 0);
         break;
      case 2:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1], 1, 0);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2], 0);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2], 0);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2], 0);
         break;
      case 3:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1],
                        matrix_i[shift_in + 2], 1);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2],
                           matrix_w[wi + 3]);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2],
                      matrix_m[wi + 3]);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2],
                      matrix_v[wi + 3]);
         break;
      default:
         inp = (float4)(matrix_i[shift_in], matrix_i[shift_in + 1],
                        matrix_i[shift_in + 2], matrix_i[shift_in + 3]);
         weight = (float4)(matrix_w[wi], matrix_w[wi + 1], matrix_w[wi + 2],
                           matrix_w[wi + 3]);
         m = (float4)(matrix_m[wi], matrix_m[wi + 1], matrix_m[wi + 2],
                      matrix_m[wi + 3]);
         v = (float4)(matrix_v[wi], matrix_v[wi + 1], matrix_v[wi + 2],
                      matrix_v[wi + 3]);
         break;
     }
   float4 g = (float4)(matrix_g[(outputs + 1) * model + i]) * inp;
   float4 mt = b1 * m + (1 - b1) * g;
   float4 vt = b2 * v + (1 - b2) * pow(g, 2);
   float4 delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   switch(min(inputs + 1 - j * 4, 3))
     {
      case 3:
         if(fabs(delta.s3) > 0)
            matrix_w[wi + 3] =
               clamp(matrix_w[wi + 3] + delta.s3, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 3] = mt.s3;
         matrix_v[wi + 3] = vt.s3;
      case 2:
         if(fabs(delta.s2) > 0)
            matrix_w[wi + 2] =
               clamp(matrix_w[wi + 2] + delta.s2, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 2] = mt.s2;
         matrix_v[wi + 2] = vt.s2;
      case 1:
         if(fabs(delta.s1) > 0)
            matrix_w[wi + 1] =
               clamp(matrix_w[wi + 1] + delta.s1, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi + 1] = mt.s1;
         matrix_v[wi + 1] = vt.s1;
      case 0:
         if(fabs(delta.s0) > 0)
            matrix_w[wi] = clamp(matrix_w[wi] + delta.s0, -MAX_WEIGHT, MAX_WEIGHT);
         matrix_m[wi] = mt.s0;
         matrix_v[wi] = vt.s0;
         break;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Concat_FeedForward(__global float *matrix_w,    ///<[in] Weights matrix (m+1)*n, where m - number
                                 ///< of neurons in layer and n - number of outputs
                                 ///< (neurons in next layer)
                                 __global float *matrix_i1, ///<[in] Inputs 1 tensor
                                 __global float *matrix_i2, ///<[in] Inputs 2 tensor
                                 __global float *matrix_o,  ///<[out] Output tensor
                                 int inputs1,               ///< Number of inputs
                                 int inputs2,               ///< Number of inputs
                                 int activation             ///< Activation type (#ENUM_ACTIVATION)
                                )
  {
   int i = get_global_id(0);
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs1 + inputs2 + 1) * i;
//---
   for(int k = 0; k < inputs1; k += 4)
     {
      switch(inputs1 - k)
        {
         case 1:
            inp = (float4)(matrix_i1[k], 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 3:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], matrix_i1[k + 2], 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], 0);
            break;
         default:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], matrix_i1[k + 2],
                           matrix_i1[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
//---
   shift += inputs1;
   for(int k = 0; k < inputs2; k += 4)
     {
      switch(inputs2 - k)
        {
         case 1:
            inp = (float4)(matrix_i2[k], 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 3:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], matrix_i2[k + 2], 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], 0);
            break;
         default:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], matrix_i2[k + 2],
                           matrix_i2[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1],
                              matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
   sum += matrix_w[shift + inputs2];
//---
   if(isnan(sum))
      sum = 0;
//---
   matrix_o[i] = Activation(sum, activation);;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Concat_HiddenGradient(__global float *matrix_w,     ///<[in] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
                                    __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
                                    __global float *matrix_o1, ///<[in] Previous layer Output tensor
                                    __global float *matrix_o2, ///<[in] Previous layer Output tensor
                                    __global float *matrix_ig1, ///<[out] Tensor of gradients at previous layer
                                    __global float *matrix_ig2, ///<[out] Tensor of gradients at previous layer
                                    int outputs,                ///< Number of outputs
                                    int inputs1, int inputs2,
                                    int activation1, ///< Activation type (#ENUM_ACTIVATION)
                                    int activation2  ///< Activation type (#ENUM_ACTIVATION)
                                   )
  {
   int i = get_global_id(0);
   if(i >= (inputs1 + inputs2))
      return;
   int inputs = inputs1 + inputs2;
   float sum = 0;
   float out = (i < inputs1 ? matrix_o1[i] : matrix_o2[i - inputs1]);
   float4 grad, weight;
   for(int k = 0; k < outputs; k += 4)
     {
      switch(outputs - k)
        {
         case 1:
            weight = (float4)(matrix_w[k * (inputs + 1) + i], 0, 0, 0);
            grad = (float4)(matrix_g[k], 0, 0, 0);
            break;
         case 2:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], 0, 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i], 0, 0);
            break;
         case 3:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2], 0);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i],
                              matrix_w[(k + 2) * (inputs + 1) + i], 0);
            break;
         default:
            grad = (float4)(matrix_g[k], matrix_g[k + 1], matrix_g[k + 2],
                            matrix_g[k + 3]);
            weight = (float4)(matrix_w[k * (inputs + 1) + i],
                              matrix_w[(k + 1) * (inputs + 1) + i],
                              matrix_w[(k + 2) * (inputs + 1) + i],
                              matrix_w[(k + 3) * (inputs + 1) + i]);
            break;
        }
      sum += dot(grad, weight);
     }
   if(isnan(sum))
      sum = 0;
   if(i < inputs1)
      matrix_ig1[i] = Deactivation(sum, out, activation1);
   else
      matrix_ig2[i - inputs1] = Deactivation(sum, out, activation2);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Concat_UpdateWeightsMomentum(__global float *matrix_w,     ///<[in,out] Weights matrix (m+1)*n, where m - number of neurons in previous layer and n - number of neurons in current layer
      __global float *matrix_g,  ///<[in] Tensor of gradients at current layer
      __global float *matrix_i1, ///<[in] Inputs tensor
      __global float *matrix_i2, ///<[in] Inputs tensor
      __global float
      *matrix_dw, ///<[in,out] Matrix of delta weights in last correction
      int inputs1,    ///< Number of inputs
      int inputs2,    ///< Number of inputs
      float learning_rates, ///< Learning rates
      float momentum        ///< Momentum multiplier
                                          )
  {
   int i = get_global_id(0);
   int j = get_global_id(1);
   if(j > (inputs1 + inputs2))
      return;
   int wi = i * (inputs1 + inputs2 + 1) + j;
   float inp = (j < inputs1 ? matrix_i1[j] : ((j - inputs1) < inputs2 ? matrix_i2[j - inputs1] : 1));
   float delta = learning_rates * matrix_g[i] * inp + momentum * matrix_dw[wi];
   if(!isnan(delta))
     {
      matrix_dw[wi] = delta;
      if(fabs(delta) > 0)
         matrix_w[wi] = clamp(matrix_w[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Concat_UpdateWeightsAdam(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                       ///< number of neurons in previous layer and n -
                                       ///< number of neurons in current layer
                                       __global const float
                                       *matrix_g, ///<[in] Tensor of gradients at current layer
                                       __global const float *matrix_i1, ///<[in] Inputs tensor
                                       __global const float *matrix_i2, ///<[in] Inputs tensor
                                       __global float *matrix_m,        ///<[in,out] Matrix of first momentum
                                       __global float *matrix_v,        ///<[in,out] Matrix of seconfd momentum
                                       const int inputs1,               ///< Number of inputs
                                       const int inputs2,               ///< Number of inputs
                                       const float l,                   ///< Learning rates
                                       const float b1,                  ///< First momentum multiplier
                                       const float b2                   ///< Second momentum multiplier
                                      )
  {
   const int i = get_global_id(0);
   const int j = get_global_id(1);
   if(j > (inputs1 + inputs2))
      return;
   const int wi = i * (inputs1 + inputs2 + 1) + j;
   float inp =
      (j < inputs1 ? matrix_i1[j]
       : ((j - inputs1) < inputs2 ? matrix_i2[j - inputs1] : 1));
   float weight = matrix_w[wi];
   float g = matrix_g[i] * inp;
   float mt = b1 * matrix_m[wi] + (1 - b1) * g;
   float vt = b2 * matrix_v[wi] + (1 - b2) * pow(g, 2);
   float delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      matrix_w[wi] = clamp(matrix_w[wi] + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[wi] = mt;
   matrix_v[wi] = vt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftUpdate(__global float *target, ///<[in,out] Target matrix
                         __global const float *source, ///<[in] Source matrix
                         const float tau ///<[in] Multiplicator Tau
                        )
  {
   const int i = get_global_id(0);
   target[i] = source[i] * tau + (1.0f - tau) * target[i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SoftUpdateAdam(__global float *target, __global const float *source,
                             __global float *matrix_m, ///<[in,out] Matrix of first momentum
                             __global float *matrix_v, ///<[in,out] Matrix of seconfd momentum
                             const float tau,          ///<[in] Multiplicator Tau
                             const float b1,           ///< First momentum multiplier
                             const float b2            ///< Second momentum multiplier
                            )
  {
   const int i = get_global_id(0);
   float m, v, weight;
   m = matrix_m[i];
   v = matrix_v[i];
   weight = target[i];
   float g = source[i] - weight;
   m = b1 * m + (1 - b1) * g;
   v = b2 * v + (1 - b2) * pow(g, 2);
   float delta = tau * m / (v != 0.0f ? sqrt(v) : 1.0f);
   if(fabs(delta) > 0)
      target[i] = clamp(weight + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[i] = m;
   matrix_v[i] = v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SAC_AlphaLogProbs(__global float *outputs,
                                __global float *quantiles,
                                __global float *probs, __global float *alphas,
                                __global float *log_probs,
                                __global float *random, const int count_quants,
                                const int activation)
  {
   const int i = get_global_id(0);
   int shift = i * count_quants;
   float prob = 0;
   float value = 0;
   float sum = 0;
   float rnd = random[i];
//---
   for(int r = 0; r < count_quants; r++)
     {
      prob = probs[shift + r];
      sum += prob;
      if(sum >= rnd || r == (count_quants - 1))
        {
         value = quantiles[shift + r];
         break;
        }
     }
//---
   outputs[i] = Activation(value, activation);
   log_probs[i] = -alphas[i] * log(prob);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SAC_AlphaGradients(__global float *outputs,
                                 __global float *gradient,
                                 __global float *log_probs,
                                 __global float *alphas_grad,
                                 const int activation)
  {
   const int i = get_global_id(0);
   float out = outputs[i];
//---
   float grad = -gradient[i] * log_probs[i];
//---
   alphas_grad[i] = Deactivation(grad, out, activation);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SAC_OutputGradient(__global float *quantiles, __global float *delta_taus,
                                 __global float *output_gr, __global float *quantiles_gr,
                                 __global float *taus_gr, __global float *output,
                                 const int count_quants, const int activation)
  {
   size_t action = get_global_id(0);
   int shift = action * count_quants;
   float quant1 = -1e37f;
   float quant2 = 1e37f;
   int pos1 = -1;
   int pos2 = -1;
   float value = output[action];
//---
   for(int i = 0; i < count_quants; i++)
     {
      float quant = Activation(quantiles[shift + i], activation);
      if(value >= quant && quant1 < quant)
        {
         quant1 = quant;
         pos1 = shift + i;
        }
      if(value < quant && quant2 > quant)
        {
         quant2 = quant;
         pos2 = shift + i;
        }
      quantiles_gr[shift + i] = 0.0f;
      taus_gr[shift + i] = 0.0f;
     }
   float gradient = output_gr[action];
   if(quant1 > -1e37f)
     {
      quantiles_gr[pos1] = gradient * delta_taus[pos1];
      taus_gr[pos1] = gradient * quant1;
     }
   if(quant2 < 1e37f)
     {
      quantiles_gr[pos2] = gradient * delta_taus[pos2];
      taus_gr[pos2] = gradient * quant2;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void SAC_CalcLogProbs(__global float *outputs,
                               __global float *quantiles, __global float *probs,
                               __global float *alphas,
                               __global float *log_probs,
                               const int count_quants, const int activation)
  {
   const int i = get_global_id(0);
   int shift = i * count_quants;
   float quant1 = -1e37f;
   float quant2 = 1e37f;
   float prob1 = 0;
   float prob2 = 0;
   float value = outputs[i];
//---
   for(int q = 0; q < count_quants; q++)
     {
      float quant = Activation(quantiles[shift + q], activation);
      if(value >= quant && quant1 < quant)
        {
         quant1 = quant;
         prob1 = probs[shift + q];
        }
      if(value < quant && quant2 > quant)
        {
         quant2 = quant;
         prob2 = probs[shift + q];
        }
     }
//---
   float prob = fabs(value - quant1) / fabs(quant2 - quant1);
   prob = clamp((1 - prob) * prob1 + prob * prob2, 1.0e-3f, 1.0f);
   log_probs[i] = -alphas[i] * log(prob);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Embedding(__global float *inputs, __global float *outputs,
                        __global float *weights, __global int *windows,
                        __global float *std, const int stack_size)
  {
   const int window_out = get_global_size(0);
   const int pos = get_local_id(0);
   const int emb = get_global_id(1);
   const int emb_total = get_global_size(1);
   const int shift_out = emb * window_out + pos;
   const int step = emb_total * window_out;
   const uint ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
//---
   for(int i = stack_size - 1; i > 0; i--)
      outputs[i * step + shift_out] = outputs[(i - 1) * step + shift_out];
   int shift_in = 0;
   for(int i = 0; i < emb; i++)
      shift_in += windows[i];
   const int shift_weights = (shift_in + emb) * window_out;
   const int window_in = windows[emb];
   const int local_pos = (pos >= ls ? pos % (ls - 1) : pos);
   const int local_orders = (window_out + ls - 1) / ls;
   const int local_order = pos / ls;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   if(local_order == 0)
      temp[local_pos] = 0;
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   float value = weights[shift_weights + window_in * window_out + pos];
   for(int i = 0; i < window_in; i++)
      value +=
         inputs[shift_in + i] * weights[shift_weights + i * window_out + pos];
   for(int i = 0; i < local_orders; i++)
     {
      if(i == local_order)
         temp[local_pos] += value;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(pos < count)
         temp[pos] += temp[pos + count];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   value -= temp[0] / (float)window_out;
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   if(local_order == 0)
      temp[local_pos] = 0;
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   for(int i = 0; i < local_orders; i++)
     {
      if(i == local_order)
         temp[local_pos] += pow(value, 2.0f) / (float)window_out;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   count = ls;
   do
     {
      count = (count + 1) / 2;
      if(pos < count)
         temp[pos] += temp[pos + count];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   if(temp[0] > 0)
      value /= sqrt(temp[0]);
//---
   outputs[shift_out] = value;
   if(pos == 0)
      std[emb] = sqrt(temp[0]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void EmbeddingHiddenGradient(__global float *inputs_gradient,
                                      __global float *outputs_gradient,
                                      __global float *weights,
                                      __global int *windows,
                                      __global float *std,
                                      const int window_out)
  {
   const int pos = get_global_id(0);
   int emb = -1;
   int count = 0;
   do
     {
      emb++;
      count += windows[emb];
     }
   while(count < pos);
   const int shift_out = emb * window_out;
   const int shift_weights = (pos + emb) * window_out;
//---
   float value = 0;
   for(int i = 0; i < window_out; i++)
      value += outputs_gradient[shift_out + i] * weights[shift_weights + i];
   float s = std[emb];
   if(s > 0)
      value /= s;
//---
   inputs_gradient[pos] = value;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void EmbeddingUpdateWeightsAdam(__global float *weights,    ///<[in,out] Weights matrix (m+1)*n, where m -
      ///< number of neurons in previous layer and n -
      ///< number of neurons in current layer
      __global const float
      *gradient,                ///<[in] Tensor of gradients at current layer
      __global const float *inputs, ///<[in] Inputs tensor
      __global float *matrix_m,     ///<[in,out] Matrix of first momentum
      __global float *matrix_v,     ///<[in,out] Matrix of seconfd momentum
      __global int *windows, __global float *std, const int window_out,
      const float l,  ///< Learning rates
      const float b1, ///< First momentum multiplier
      const float b2  ///< Second momentum multiplier
                                        )
  {
   const int i = get_global_id(0);
   int emb = -1;
   int count = 0;
   int shift = 0;
   do
     {
      emb++;
      shift = count;
      count += (windows[emb] + 1) * window_out;
     }
   while(count <= i);
   const int shift_out = emb * window_out;
   int shift_in = shift / window_out - emb;
   shift = (i - shift) / window_out;
   float inp = 1.0f;
   if(shift < windows[emb])
      inp = inputs[shift_in + shift];
//---
   float weight = weights[i];
   float g = gradient[shift_out] * inp / std[emb];
   float mt = b1 * matrix_m[i] + (1 - b1) * g;
   float vt = b2 * matrix_v[i] + (1 - b2) * pow(g, 2);
   float delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      weights[i] = clamp(weights[i] + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[i] = mt;
   matrix_v[i] = vt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void Transpose(__global float *matrix_in, ///<[in] Input matrix
                        __global float *matrix_out ///<[out] Output matrix
                       )
  {
   const int r = get_global_id(0);
   const int c = get_global_id(1);
   const int rows = get_global_size(0);
   const int cols = get_global_size(1);
//---
   matrix_out[c * rows + r] = matrix_in[r * cols + c];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MH2AttentionOut(__global float *q,     ///<[in] Matrix of Querys
                              __global float *kv,    ///<[in] Matrix of Keys
                              __global float *score, ///<[out] Matrix of Scores
                              __global float *out, ///<[out] Matrix of attention
                              int dimension,        ///< Dimension of Key
                              int heads_kv,
                              int mask ///< 1 - calc only previous units, 0 - calc all
                             )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_k = dimension * (2 *  heads_kv * k + h_kv);
   const int shift_v = dimension * (2 *  heads_kv * k + heads_kv + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- sum of exp
   uint count = 0;
   if(k < ls)
     {
      temp[k] = 0;
      do
        {
         if(mask == 0 || q_id >= (count * ls + k))
            if((count * ls) < (kunits - k))
              {
               float sum = 0;
               int sh_k = 2 * dimension * heads_kv * count * ls;
               for(int d = 0; d < dimension; d++)
                  sum = q[shift_q + d] * kv[shift_k + d + sh_k];
               sum = exp(sum / koef);
               if(isnan(sum))
                  sum = 0;
               temp[k] = temp[k] + sum;
              }
         count++;
        }
      while((count * ls + k) < kunits);
     }
   barrier(CLK_LOCAL_MEM_FENCE);
   count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k < ls)
         temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
      if(k + count < ls)
         temp[k + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- score
   float sum = temp[0];
   float sc = 0;
   if(mask == 0 || q_id >= (count * ls + k))
      if(sum != 0)
        {
         for(int d = 0; d < dimension; d++)
            sc = q[shift_q + d] * kv[shift_k + d];
         sc = exp(sc / koef) / sum;
         if(isnan(sc))
            sc = 0;
        }
   score[shift_s] = sc;
   barrier(CLK_LOCAL_MEM_FENCE);
//--- out
   for(int d = 0; d < dimension; d++)
     {
      uint count = 0;
      if(k < ls)
         do
           {
            if((count * ls) < (kunits - k))
              {
               int sh_v = 2 * dimension * heads_kv * count * ls;
               float sum =
                  kv[shift_v + d + sh_v] * (count == 0 ? sc : score[shift_s + count * ls]);
               if(isnan(sum))
                  sum = 0;
               temp[k] = (count > 0 ? temp[k] : 0) + sum;
              }
            count++;
           }
         while((count * ls + k) < kunits);
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k < ls)
            temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
         if(k + count < ls)
            temp[k + count] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      out[shift_q + d] = temp[0];
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MH2AttentionInsideGradients(__global float *q, __global float *q_g,
      __global float *kv, __global float *kv_g,
      __global float *scores, __global float *gradient,
      int kunits, int heads_kv)
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = q_id * kunits * heads + h * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
      for(int v = q_id; v < kunits; v += qunits)
        {
         float grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v;
            for(int g = 0; g < qunits; g++)
               grad += gradient[shift_g + dimension * (hq - h + g  * heads)] *
                       scores[shift_score + g * step_score];
           }
         int shift_v = dimension * (2 *  heads_kv * v + heads_kv + h) + d;
         kv_g[shift_v] = grad;
        }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = (heads_kv + h_kv) * dimension + d;
   int shift_key = h_kv * dimension + d;
   for(int k = 0; k < kunits; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k];
      if(sc == 0)
         continue;
      for(int v = 0; v < kunits; v++)
         sc_g += scores[shift_s + v] * out_g * kv[shift_val + 2 * v * heads_kv * dimension] *
                 ((float)(k == v) - sc);
      grad += sc_g * kv[shift_key + 2 * k * heads_kv * dimension];
     }
   q_g[shift_q] = grad / koef;
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      for(int k = q_id; k < kunits; k += qunits)
        {
         int shift_k = dimension * (2 *  heads_kv * k + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq++)
           {
            int shift_score = hq * kunits + k;
            float val = kv[shift_k + heads_kv * dimension];
            for(int scr = 0; scr < qunits; scr++)
              {
               float sc_g = 0;
               int shift_sc = scr * kunits * heads;
               float sc = scores[shift_sc + k];
               if(sc == 0)
                  continue;
               for(int v = 0; v < kunits; v++)
                  sc_g += scores[shift_sc + v] * gradient[shift_g + scr * dimension] *
                          val * ((float)(k == v) - sc);
               grad += sc_g * q[shift_q + scr * dimension];
              }
           }
         kv_g[shift_k] = grad / koef;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CGConv_HiddenGradient(__global const float *matrix_g,     ///<[in] Tensor of gradients at current layer
                                    __global const float *matrix_f,  ///<[in] Previous layer Output tensor
                                    __global const float *matrix_s,  ///<[in] Previous layer Output tensor
                                    __global float *matrix_fg, ///<[out] Tensor of gradients at previous layer
                                    __global float *matrix_sg, ///<[out] Tensor of gradients at previous layer
                                    const int activationf,           ///< Activation type (#ENUM_ACTIVATION)
                                    const int activations            ///< Activation type (#ENUM_ACTIVATION)
                                   )
  {
   int i = get_global_id(0);
//---
   float grad = matrix_g[i];
   float f = matrix_f[i];
   float s = matrix_s[i];
//---
   float sg = grad * f;
   float fg = grad * s;
//---
   matrix_fg[i] = Deactivation(fg, f, activationf);
   matrix_sg[i] = Deactivation(sg, s, activations);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void XCiTFeedForward(__global float *qkv, __global float *score,
                              __global float *out)
  {
   const size_t d = get_local_id(0);
   const size_t dimension = get_local_size(0);
   const size_t u = get_local_id(1);
   const size_t units = get_local_size(1);
   const size_t h = get_global_id(2);
   const size_t heads = get_global_size(2);
//---
   const uint ls_u = min((uint)units, (uint)LOCAL_ARRAY_SIZE);
   const uint ls_d = min((uint)dimension, (uint)LOCAL_ARRAY_SIZE);
   __local float q[LOCAL_ARRAY_SIZE][LOCAL_ARRAY_SIZE];
   __local float k[LOCAL_ARRAY_SIZE][LOCAL_ARRAY_SIZE];
//--- Normalize Query and Key
   for(int cur_d = 0; cur_d < dimension; cur_d += ls_d)
     {
      float q_val = 0;
      float k_val = 0;
      //---
      if(d < ls_d && (cur_d + d) < dimension && u < ls_u)
        {
         for(int count = u; count < units; count += ls_u)
           {
            int shift = count * dimension * heads * 3 + dimension * h + cur_d + d;
            q_val += pow(qkv[shift], 2.0f);
            k_val += pow(qkv[shift + dimension * heads], 2.0f);
           }
         q[u][d] = q_val;
         k[u][d] = k_val;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      uint count = ls_u;
      do
        {
         count = (count + 1) / 2;
         if(d < ls_d)
           {
            if(u < ls_u && u < count && (u + count) < units)
              {
               float q_val = q[u][d] + q[u + count][d];
               float k_val = k[u][d] + k[u + count][d];
               q[u + count][d] = 0;
               k[u + count][d] = 0;
               q[u][d] = q_val;
               k[u][d] = k_val;
              }
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      int shift = u * dimension * heads * 3 + dimension * h + cur_d;
      qkv[shift] = qkv[shift] / sqrt(q[0][d]);
      qkv[shift + dimension * heads] =
         qkv[shift + dimension * heads] / sqrt(k[0][d]);
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//--- Score
   int step = dimension * heads * 3;
   for(int cur_r = 0; cur_r < dimension; cur_r += ls_u)
     {
      for(int cur_d = 0; cur_d < dimension; cur_d += ls_d)
        {
         if(u < ls_d && d < ls_d)
            q[u][d] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
         //---
         if((cur_r + u) < ls_d && (cur_d + d) < ls_d)
           {
            int shift_q = dimension * h + cur_d + d;
            int shift_k = dimension * (heads + h) + cur_r + u;
            float scr = 0;
            for(int i = 0; i < units; i++)
               scr += qkv[shift_q + i * step] * qkv[shift_k + i * step];
            scr = exp(scr / sqrt((float)units));
            score[(cur_r + u) * dimension * heads + dimension * h + cur_d + d] =
               scr;
            q[u][d] += scr;
           }
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      int count = ls_d;
      do
        {
         count = (count + 1) / 2;
         if(u < ls_d)
           {
            if(d < ls_d && d < count && (d + count) < dimension)
               q[u][d] += q[u][d + count];
            if(d + count < ls_d)
               q[u][d + count] = 0;
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      if((cur_r + u) < ls_d)
         score[(cur_r + u) * dimension * heads + dimension * h + d] /= q[u][0];
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   int shift_out = dimension * (u * heads + h) + d;
   int shift_s = dimension * (heads * d + h);
   int shift_v = dimension * (heads * (u * 3 + 2) + h);
   float sum = 0;
   for(int i = 0; i < dimension; i++)
      sum += qkv[shift_v + i] * score[shift_s + i];
   out[shift_out] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void XCiTInsideGradients(__global float *qkv, __global float *qkv_g,
                                  __global float *scores,
                                  __global float *gradient)
  {
//--- init
   const int q = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int units = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (heads * 3 * q + h);
   const int shift_k = dimension * (heads * (3 * q + 1) + h);
   const int shift_v = dimension * (heads * (3 * q + 2) + h);
   const int shift_g = dimension * (heads * q + h);
   int shift_score = dimension * h;
   int step_score = dimension * heads;
//--- Calculating Value's gradients
   float sum = 0;
   for(int i = 0; i < dimension; i++)
      sum += gradient[shift_g + i] * scores[shift_score + d + i * step_score];
   qkv_g[shift_v + d] = sum;
//--- Calculating Query's gradients
   float grad = 0;
   float val = qkv[shift_v + d];
   for(int k = 0; k < dimension; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_score + k];
      for(int v = 0; v < dimension; v++)
         sc_g += scores[shift_score + v] * val *
                 gradient[shift_g + v * dimension] * ((float)(k == v) - sc);
      grad += sc_g * qkv[shift_k + k];
     }
   qkv_g[shift_q + d] = grad / sqrt((float)units);
//--- Calculating Key's gradients
   grad = 0;
   float out_g = gradient[shift_g];
   for(int scr = 0; scr < dimension; scr++)
     {
      float sc_g = 0;
      int shift_sc = scr * dimension * heads;
      float sc = scores[shift_sc + d];
      for(int v = 0; v < dimension; v++)
         sc_g += scores[shift_sc + v] * out_g * qkv[shift_v + v] *
                 ((float)(d == v) - sc);
      grad += sc_g * qkv[shift_q + scr];
     }
   qkv_g[shift_k + d] = grad / sqrt((float)units);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void DOTFeedForward(__global float *qkv, __global float *score,
                             __global float *rpb, __global float *out)
  {
   const size_t d = get_local_id(0);
   const size_t dimension = get_local_size(0);
   const size_t u = get_global_id(1);
   const size_t units = get_global_size(1);
   const size_t h = get_global_id(2);
   const size_t heads = get_global_size(2);
//---
   uint step = 3 * dimension * heads;
   uint start = max((int)u - 1, 0);
   uint stop = min((int)u + 1, (int)units - 1);
   uint shift_q = u * step + h * dimension;
   uint shift_k = start * step + dimension * (heads + h);
   uint shift_score = u * 3 * heads;
//---
   const uint ls_d = min((uint)dimension, (uint)LOCAL_ARRAY_SIZE);
   __local float temp[LOCAL_ARRAY_SIZE][3];
//--- Score
   if(d < ls_d)
     {
      for(uint pos = start; pos <= stop; pos++)
        {
         temp[d][pos - start] = 0;
        }
      for(uint dim = d; dim < dimension; dim += ls_d)
        {
         float q = qkv[shift_q + dim];
         for(uint pos = start; pos <= stop; pos++)
           {
            uint i = pos - start;
            temp[d][i] = temp[d][i] + q * qkv[shift_k + i * step + dim];
           }
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      int count = ls_d;
      do
        {
         count = (count + 1) / 2;
         if(d < count && (d + count) < dimension)
            for(uint i = 0; i <= (stop - start); i++)
              {
               temp[d][i] += temp[d + count][i];
               temp[d + count][i] = 0;
              }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
     }
//---
   if(d == 0)
     {
      float sum = 0;
      for(uint i = 0; i <= (stop - start); i++)
        {
         temp[0][i] = exp(temp[0][i] + rpb[shift_score + i]);
         sum += temp[0][i];
        }
      for(uint i = 0; i <= (stop - start); i++)
        {
         temp[0][i] = temp[0][i] / sum;
         score[shift_score + i] = temp[0][i];
        }
     }
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   int shift_out = dimension * (u * heads + h) + d;
   int shift_v = dimension * (heads * (u * 3 + 2) + h);
   float sum = 0;
   for(uint i = 0; i <= (stop - start); i++)
      sum += qkv[shift_v + i] * temp[0][i];
   out[shift_out] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void DOTInsideGradients(__global float *qkv, __global float *qkv_g,
                                 __global float *scores, __global float *rpb,
                                 __global float *rpb_g,
                                 __global float *gradient)
  {
//--- init
   const uint u = get_global_id(0);
   const uint d = get_global_id(1);
   const uint h = get_global_id(2);
   const uint units = get_global_size(0);
   const uint dimension = get_global_size(1);
   const uint heads = get_global_size(2);
//---
   uint step = 3 * dimension * heads;
   uint start = max((int)u - 1, 0);
   uint stop = min((int)u + 1, (int)units - 1);
   const uint shift_q = u * step + dimension * h + d;
   const uint shift_k = u * step + dimension * (heads + h) + d;
   const uint shift_v = u * step + dimension * (2 * heads + h) + d;
//--- Calculating Value's gradients
   float sum = 0;
   for(uint i = start; i <= stop; i++)
     {
      int shift_score = i * 3 * heads;
      if(u == i)
        {
         shift_score += (uint)(u > 0);
        }
      else
        {
         if(u > i)
            shift_score += (uint)(start > 0) + 1;
        }
      uint shift_g = dimension * (i * heads + h) + d;
      sum += gradient[shift_g] * scores[shift_score];
     }
   qkv_g[shift_v] = sum;
//--- Calculating Query's gradients
   float grad = 0;
   uint shift_score = u * heads * 3;
   for(int k = start; k <= stop; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_score + k - start];
      for(int v = start; v <= stop; v++)
         for(int dim = 0; dim < dimension; dim++)
            sc_g += scores[shift_score + v - start] *
                    qkv[v * step + dimension * (2 * heads + h) + dim] *
                    gradient[dimension * (u * heads + h) + dim] *
                    ((float)(k == v) - sc);
      grad += sc_g * qkv[k * step + dimension * (heads + h) + d];
      if(d == 0)
         rpb_g[shift_score + k - start] = sc_g;
     }
   qkv_g[shift_q] = grad;
//--- Calculating Key's gradients
   grad = 0;
   for(int q = start; q <= stop; q++)
     {
      float sc_g = 0;
      shift_score = q * heads * 3;
      if(u == q)
        {
         shift_score += (uint)(u > 0);
        }
      else
        {
         if(u > q)
            shift_score += (uint)(start > 0) + 1;
        }
      float sc = scores[shift_score];
      for(int v = start; v <= stop; v++)
        {
         shift_score = v * heads * 3;
         if(u == v)
           {
            shift_score += (uint)(u > 0);
           }
         else
           {
            if(u > v)
               shift_score += (uint)(start > 0) + 1;
           }
         for(int dim = 0; dim < dimension; dim++)
            sc_g += scores[shift_score] * qkv[shift_v - d + dim] *
                    gradient[dimension * (v * heads + h) + d] *
                    ((float)(d == v) - sc);
        }
      grad += sc_g * qkv[q * step + dimension * h + d];
     }
   qkv_g[shift_k] = grad;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void RPBUpdateAdam(__global float *target, __global const float *gradient,
                            __global float *matrix_m, ///<[in,out] Matrix of first momentum
                            __global float *matrix_v, ///<[in,out] Matrix of seconfd momentum
                            const float b1,           ///< First momentum multiplier
                            const float b2            ///< Second momentum multiplier
                           )
  {
   const int i = get_global_id(0);
   float m, v, weight;
   m = matrix_m[i];
   v = matrix_v[i];
   weight = target[i];
   float g = gradient[i];
   m = b1 * m + (1 - b1) * g;
   v = b2 * v + (1 - b2) * pow(g, 2);
   float delta = m / (v != 0.0f ? sqrt(v) : 1.0f);
   target[i] = clamp(weight + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[i] = m;
   matrix_v[i] = v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void GTEFeedForward(__global float *qkv, __global float *score,
                             __global float *out, int dimension)
  {
   const size_t cur_q = get_global_id(0);
   const size_t units_q = get_global_size(0);
   const size_t cur_k = get_local_id(1);
   const size_t units_k = get_local_size(1);
   const size_t h = get_global_id(2);
   const size_t heads = get_global_size(2);
//---
   int shift_q = dimension * (cur_q + h * units_q);
   int shift_k = (cur_k + h * units_k + heads * units_q);
   int shift_v = dimension * (h * units_k + heads * (units_q + units_k));
   int shift_score_con = units_k * (cur_q * 2 * heads + h) + cur_k;
   int shift_score_notcon = units_k * (cur_q * 2 * heads + heads + h) + cur_k;
   int shift_out_con = dimension * (cur_q + h * units_q);
   int shift_out_notcon = dimension * (cur_q + units_q * (h + heads));
//---
   const uint ls_score = min((uint)units_k, (uint)LOCAL_ARRAY_SIZE);
   __local float local_score[LOCAL_ARRAY_SIZE][2];
//--- Score
   float scr = 0;
   for(int d = 0; d < dimension; d++)
      scr += qkv[shift_q + d] * qkv[shift_k + d];
   scr = exp(min(scr / sqrt((float)dimension), 30.0f));
   if(cur_q == cur_k)
     {
      score[shift_score_con] = scr;
      score[shift_score_notcon] = scr;
      if(cur_k < ls_score)
        {
         local_score[cur_k][0] = scr;
         local_score[cur_k][1] = scr;
        }
     }
   else
     {
      if(abs(cur_q - cur_k) == 1)
        {
         score[shift_score_con] = scr;
         score[shift_score_notcon] = 0;
         if(cur_k < ls_score)
           {
            local_score[cur_k][0] = scr;
            local_score[cur_k][1] = 0;
           }
        }
      else
        {
         score[shift_score_con] = 0;
         score[shift_score_notcon] = scr;
         if(cur_k < ls_score)
           {
            local_score[cur_k][0] = 0;
            local_score[cur_k][1] = scr;
           }
        }
     }
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   for(int k = ls_score; k < units_k; k += ls_score)
     {
      if((cur_k + k) < units_k)
        {
         local_score[cur_k][0] += score[shift_score_con + k];
         local_score[cur_k][1] += score[shift_score_notcon + k];
        }
     }
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   int count = ls_score;
   do
     {
      count = (count + 1) / 2;
      if(cur_k < count)
        {
         if((cur_k + count) < units_k)
           {
            local_score[cur_k][0] += local_score[cur_k + count][0];
            local_score[cur_k][1] += local_score[cur_k + count][1];
            local_score[cur_k + count][0] = 0;
            local_score[cur_k + count][1] = 0;
           }
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   score[shift_score_con] /= local_score[0][0];
   score[shift_score_notcon] /= local_score[0][1];
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   shift_score_con -= cur_k;
   shift_score_notcon -= cur_k;
   for(int d = 0; d < dimension; d += ls_score)
     {
      if((cur_k + d) < dimension)
        {
         float sum_con = 0;
         float sum_notcon = 0;
         for(int v = 0; v < units_k; v++)
           {
            sum_con += qkv[shift_v + v * dimension + cur_k + d] *
                       score[shift_score_con + v];
            sum_notcon += qkv[shift_v + v * dimension + cur_k + d] *
                          score[shift_score_notcon + v];
           }
         out[shift_out_con + cur_k + d] = sum_con;
         out[shift_out_notcon + cur_k + d] = sum_notcon;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void GTEInsideGradients(__global float *qkv, __global float *qkv_g,
                                 __global float *scores,
                                 __global float *gradient)
  {
//--- init
   const uint u = get_global_id(0);
   const uint d = get_global_id(1);
   const uint h = get_global_id(2);
   const uint units = get_global_size(0);
   const uint dimension = get_global_size(1);
   const uint heads = get_global_size(2);
//--- Calculating Value's gradients
     {
      int shift_out_con = dimension * h * units + d;
      int shift_out_notcon = dimension * units * (h + heads) + d;
      int shift_score_con = units * h + u;
      int shift_score_notcon = units * (heads + h) + u;
      int step_score = units * 2 * heads;
      int shift_v = dimension * (h * units + 2 * heads * units + u) + d;
      //---
      float sum = 0;
      for(uint i = 0; i <= units; i++)
        {
         sum += gradient[shift_out_con + i * dimension] *
                scores[shift_score_con + i * step_score];
         sum += gradient[shift_out_notcon + i * dimension] *
                scores[shift_score_notcon + i * step_score];
        }
      qkv_g[shift_v] = sum;
     }
//--- Calculating Query's gradients
     {
      int shift_q = dimension * (u + h * units) + d;
      int shift_out_con = dimension * (h * units + u) + d;
      int shift_out_notcon = dimension * (u + units * (h + heads)) + d;
      int shift_score_con = units * h;
      int shift_score_notcon = units * (heads + h);
      int shift_v = dimension * (h * units + 2 * heads * units);
      float grad = 0;
      for(int k = 0; k < units; k++)
        {
         int shift_k = (k + h * units + heads * units) + d;
         float sc_g = 0;
         float sc_con = scores[shift_score_con + k];
         float sc_notcon = scores[shift_score_notcon + k];
         for(int v = 0; v < units; v++)
            for(int dim = 0; dim < dimension; dim++)
              {
               sc_g += scores[shift_score_con + v] *
                       qkv[shift_v + v * dimension + dim] *
                       gradient[shift_out_con + dim] * ((float)(k == v) - sc_con);
               sc_g += scores[shift_score_notcon + v] *
                       qkv[shift_v + v * dimension + dim] *
                       gradient[shift_out_notcon + dim] *
                       ((float)(k == v) - sc_notcon);
              }
         grad += sc_g * qkv[shift_k];
        }
      qkv_g[shift_q] = grad;
     }
//--- Calculating Key's gradients
     {
      int shift_k = (u + (h + heads) * units) + d;
      int shift_out_con = dimension * h * units + d;
      int shift_out_notcon = dimension * units * (h + heads) + d;
      int shift_score_con = units * h + u;
      int shift_score_notcon = units * (heads + h) + u;
      int step_score = units * 2 * heads;
      int shift_v = dimension * (h * units + 2 * heads * units);
      float grad = 0;
      for(int q = 0; q < units; q++)
        {
         int shift_q = dimension * (q + h * units) + d;
         float sc_g = 0;
         float sc_con = scores[shift_score_con + u + q * step_score];
         float sc_notcon = scores[shift_score_notcon + u + q * step_score];
         for(int g = 0; g < units; g++)
           {
            for(int dim = 0; dim < dimension; dim++)
              {
               sc_g += scores[shift_score_con + g] *
                       qkv[shift_v + u * dimension + dim] *
                       gradient[shift_out_con + g * dimension + dim] *
                       ((float)(u == g) - sc_con);
               sc_g += scores[shift_score_notcon + g] *
                       qkv[shift_v + u * dimension + dim] *
                       gradient[shift_out_notcon + g * dimension + dim] *
                       ((float)(u == g) - sc_notcon);
              }
           }
         grad += sc_g * qkv[shift_q];
        }
      qkv_g[shift_k] = grad;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForwardNODEF(__global float *matrix_w,    ///<[in] Weights matrix (m+1)*n, where m - input
                               ///< window and n - output window
                               __global float *matrix_i, ///<[in] Inputs tensor
                               __global float *matrix_o, ///<[out] Output tensor
                               int dimension,            ///< input dimension
                               float step,               ///< h
                               int activation            ///< Activation type (#ENUM_ACTIVATION)
                              )
  {
   int d = get_global_id(0);
   int dimension_out = get_global_size(0);
   int v = get_global_id(1);
   int variables = get_global_size(1);
   int i = get_global_id(2);
   int lenth = get_global_size(2);
//---
   int shift = variables * i + v;
   int input_shift = shift * dimension;
   int output_shift = shift * dimension_out + d;
   int weight_shift = (v * dimension_out + d) * (dimension + 2);
//---
   float sum = matrix_w[dimension + 1 + weight_shift] +
               matrix_w[dimension + weight_shift] * step;
   for(int w = 0; w < dimension; w++)
      sum += matrix_w[w + weight_shift] * matrix_i[input_shift + w];
//---
   if(isnan(sum))
      sum = 0;
//---
   matrix_o[output_shift] = Activation(sum, activation);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForwardNODEInpK(__global float *matrix_i,    ///<[in] Inputs tensor
                                  __global float *matrix_k1,   ///<[in] K1 tensor
                                  __global float *matrix_k2,   ///<[in] K2 tensor
                                  __global float *matrix_k3,   ///<[in] K3 tensor
                                  __global float *matrix_k4,   ///<[in] K4 tensor
                                  __global float *matrix_k5,   ///<[in] K5 tensor
                                  __global float *matrix_k6,   ///<[in] K6 tensor
                                  __global float *matrix_beta, ///<[in] beta tensor
                                  __global float *matrix_o     ///<[out] Output tensor
                                 )
  {
   int i = get_global_id(0);
//---
   float sum = matrix_i[i];
   for(int b = 0; b < 6; b++)
     {
      float beta = matrix_beta[b];
      if(beta == 0.0f || isnan(beta))
         continue;
      //---
      float val = 0.0f;
      switch(b)
        {
         case 0:
            val = matrix_k1[i];
            break;
         case 1:
            val = matrix_k2[i];
            break;
         case 2:
            val = matrix_k3[i];
            break;
         case 3:
            val = matrix_k4[i];
            break;
         case 4:
            val = matrix_k5[i];
            break;
         case 5:
            val = matrix_k6[i];
            break;
        }
      if(val == 0.0f || isnan(val))
         continue;
      //---
      sum += val * beta;
     }
//---
   matrix_o[i] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void HiddenGradientNODEInpK(__global float *matrix_ig,   ///<[in] Inputs tensor
                                     __global float *matrix_k1g,  ///<[in] K1 tensor
                                     __global float *matrix_k2g,  ///<[in] K2 tensor
                                     __global float *matrix_k3g,  ///<[in] K3 tensor
                                     __global float *matrix_k4g,  ///<[in] K4 tensor
                                     __global float *matrix_k5g,  ///<[in] K5 tensor
                                     __global float *matrix_k6g,  ///<[in] K6 tensor
                                     __global float *matrix_beta, ///<[in] beta tensor
                                     __global float *matrix_og    ///<[out] Output tensor
                                    )
  {
   int i = get_global_id(0);
//---
   float grad = matrix_og[i];
   matrix_ig[i] = grad;
   for(int b = 0; b < 6; b++)
     {
      float beta = matrix_beta[b];
      if(isnan(beta))
         beta = 0.0f;
      //---
      float val = beta * grad;
      if(isnan(val))
         val = 0.0f;
      switch(b)
        {
         case 0:
            matrix_k1g[i] = val;
            break;
         case 1:
            matrix_k2g[i] = val;
            break;
         case 2:
            matrix_k3g[i] = val;
            break;
         case 3:
            matrix_k4g[i] = val;
            break;
         case 4:
            matrix_k5g[i] = val;
            break;
         case 5:
            matrix_k6g[i] = val;
            break;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void HiddenGradientNODEF(__global float *matrix_w,     ///<[in] Weights matrix (m+1)*n, where m - input
                                  ///< window and n - output window
                                  __global float *matrix_g,  ///<[in] Gradient tensor
                                  __global float *matrix_i,  ///<[in] Inputs tensor
                                  __global float *matrix_ig, ///<[out] Inputs Gradient tensor
                                  int dimension_out,         ///< output dimension
                                  int activation             ///< Input Activation type (#ENUM_ACTIVATION)
                                 )
  {
   int d = get_global_id(0);
   int dimension = get_global_size(0);
   int v = get_global_id(1);
   int variables = get_global_size(1);
   int i = get_global_id(2);
   int lenth = get_global_size(2);
//---
   int shift = variables * i + v;
   int input_shift = shift * dimension + d;
   int output_shift = shift * dimension_out;
   int weight_step = (dimension + 2);
   int weight_shift = (v * dimension_out) * weight_step + d;
//---
   float sum = 0;
   for(int k = 0; k < dimension_out; k++)
      sum +=
         matrix_g[output_shift + k] * matrix_w[weight_shift + k * weight_step];
   if(isnan(sum))
      sum = 0;
//---
   float out = matrix_i[input_shift];
//---
   matrix_ig[input_shift] = Deactivation(sum, out, activation);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void NODEF_UpdateWeightsAdam(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                      ///< number of neurons in previous layer and n -
                                      ///< number of neurons in current layer
                                      __global const float *matrix_gk1, ///<[in] Tensor of gradients at k1
                                      __global const float *matrix_gk2, ///<[in] Tensor of gradients at k2
                                      __global const float *matrix_gk3, ///<[in] Tensor of gradients at k3
                                      __global const float *matrix_gk4, ///<[in] Tensor of gradients at k4
                                      __global const float *matrix_gk5, ///<[in] Tensor of gradients at k5
                                      __global const float *matrix_gk6, ///<[in] Tensor of gradients at k6
                                      __global const float *matrix_ik1, ///<[in] Inputs tensor
                                      __global const float *matrix_ik2, ///<[in] Inputs tensor
                                      __global const float *matrix_ik3, ///<[in] Inputs tensor
                                      __global const float *matrix_ik4, ///<[in] Inputs tensor
                                      __global const float *matrix_ik5, ///<[in] Inputs tensor
                                      __global const float *matrix_ik6, ///<[in] Inputs tensor
                                      __global float *matrix_m,         ///<[in,out] Matrix of first momentum
                                      __global float *matrix_v,         ///<[in,out] Matrix of seconfd momentum
                                      __global const float *alpha,      ///< h
                                      const int lenth,                  ///< Number of inputs
                                      const float l,                    ///< Learning rates
                                      const float b1,                   ///< First momentum multiplier
                                      const float b2                    ///< Second momentum multiplier
                                     )
  {
   const int d_in = get_global_id(0);
   const int dimension_in = get_global_size(0);
   const int d_out = get_global_id(1);
   const int dimension_out = get_global_size(1);
   const int v = get_global_id(2);
   const int variables = get_global_id(2);
//---
   const int weight_shift = (v * dimension_out + d_out) * dimension_in;
   const int input_step = variables * (dimension_in - 2);
   const int input_shift = v * (dimension_in - 2) + d_in;
   const int output_step = variables * dimension_out;
   const int output_shift = v * dimension_out + d_out;
//---
   float weight = matrix_w[weight_shift];
   float g = 0;
   for(int i = 0; i < lenth; i++)
     {
      int shift_g = i * output_step + output_shift;
      int shift_i = i * input_step + input_shift;
      switch(dimension_in - d_in)
        {
         case 1:
            g += matrix_gk1[shift_g] + matrix_gk2[shift_g] + matrix_gk3[shift_g] +
                 matrix_gk4[shift_g] + matrix_gk5[shift_g] + matrix_gk6[shift_g];
            break;
         case 2:
            g += matrix_gk1[shift_g] * alpha[0] + matrix_gk2[shift_g] * alpha[1] +
                 matrix_gk3[shift_g] * alpha[2] + matrix_gk4[shift_g] * alpha[3] +
                 matrix_gk5[shift_g] * alpha[4] + matrix_gk6[shift_g] * alpha[5];
            break;
         default:
            g += matrix_gk1[shift_g] * matrix_ik1[shift_i] +
                 matrix_gk2[shift_g] * matrix_ik2[shift_i] +
                 matrix_gk3[shift_g] * matrix_ik3[shift_i] +
                 matrix_gk4[shift_g] * matrix_ik4[shift_i] +
                 matrix_gk5[shift_g] * matrix_ik5[shift_i] +
                 matrix_gk6[shift_g] * matrix_ik6[shift_i];
            break;
        }
     }
//---
   float mt = b1 * matrix_m[weight_shift] + (1 - b1) * g;
   float vt = b2 * matrix_v[weight_shift] + (1 - b2) * pow(g, 2);
   float delta =
      l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      matrix_w[weight_shift] =
         clamp(matrix_w[weight_shift] + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[weight_shift] = mt;
   matrix_v[weight_shift] = vt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void TimeDerivative(__global float *qkv, __global float *dqkv,
                             int dimension)
  {
   const size_t pos = get_global_id(0);
   const size_t variable = get_global_id(1);
   const size_t head = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
   const size_t heads = get_global_size(2);
//---
   const int shift = 3 * heads * variables * dimension;
   const int shift_query =
      pos * shift + (3 * variable * heads + head) * dimension;
   const int shift_key = shift_query + heads * dimension;
//---
   for(int i = 0; i < dimension; i++)
     {
      //--- dQ/dt
        {
         int count = 0;
         float delta = 0;
         float value = qkv[shift_query + i];
         if(pos > 0)
           {
            delta = value - qkv[shift_query + i - shift];
            count++;
           }
         if(pos < (total - 1))
           {
            delta += qkv[shift_query + i + shift] - value;
            count++;
           }
         if(count > 0)
            dqkv[shift_query + i] = delta / count;
        }
      //--- dK/dt
        {
         int count = 0;
         float delta = 0;
         float value = qkv[shift_key + i];
         if(pos > 0)
           {
            delta = value - qkv[shift_key + i - shift];
            count++;
           }
         if(pos < (total - 1))
           {
            delta += qkv[shift_key + i + shift] - value;
            count++;
           }
         if(count > 0)
            dqkv[shift_key + i] = delta / count;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void HiddenGradientTimeDerivative(__global float *qkv_g,
      __global float *dqkv_g,
      int dimension)
  {
   const size_t pos = get_global_id(0);
   const size_t variable = get_global_id(1);
   const size_t head = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
   const size_t heads = get_global_size(2);
//---
   const int shift = 3 * heads * variables * dimension;
   const int shift_query =
      pos * shift + (3 * variable * heads + head) * dimension;
   const int shift_key = shift_query + heads * dimension;
//---
   for(int i = 0; i < dimension; i++)
     {
      //--- dQ/dt
        {
         int count = 0;
         float grad = 0;
         float current = dqkv_g[shift_query + i];
         if(pos > 0)
           {
            grad += current - dqkv_g[shift_query + i - shift];
            count++;
           }
         if(pos < (total - 1))
           {
            grad += dqkv_g[shift_query + i + shift] - current;
            count++;
           }
         if(count > 0)
            grad /= count;
         qkv_g[shift_query + i] += grad;
        }
      //--- dK/dt
        {
         int count = 0;
         float grad = 0;
         float current = dqkv_g[shift_key + i];
         if(pos > 0)
           {
            grad += current - dqkv_g[shift_key + i - shift];
            count++;
           }
         if(pos < (total - 1))
           {
            grad += dqkv_g[shift_key + i + shift] - current;
            count++;
           }
         if(count > 0)
            grad /= count;
         qkv_g[shift_key + i] += dqkv_g[shift_key + i] + grad;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForwardContAtt(__global float *qkv, __global float *dqkv,
                                 __global float *score, __global float *out,
                                 int dimension,
                                 int heads)
  {
   const size_t query = get_global_id(0);
   const size_t key = get_global_id(1);
   const size_t variable = get_global_id(2);
   const size_t queris = get_global_size(0);
   const size_t keis = get_global_size(1);
   const size_t variables = get_global_size(2);
//---
   const uint ls_score = min((uint)keis, (uint)LOCAL_ARRAY_SIZE);
   __local float local_score[LOCAL_ARRAY_SIZE];
//---
   for(int head = 0; head < heads; head++)
     {
      const int shift = 3 * heads * variables * dimension;
      const int shift_query =
         query * shift + (3 * variable * heads + head) * dimension;
      const int shift_key =
         key * shift + (3 * variable * heads + heads + head) * dimension;
      const int shift_out =
         dimension * (heads * (query * variables + variable) + head);
      int shift_score = keis * (heads * (query * variables + variable) + head) + key;
      //--- Score
      float scr = 0;
      for(int d = 0; d < dimension; d++)
         scr += qkv[shift_query + d] * dqkv[shift_key + d] +
                qkv[shift_key + d] * dqkv[shift_query + d];
      scr = exp(min(scr / sqrt((float)dimension), 30.0f));
      score[shift_score] = scr;
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      if(key < ls_score)
        {
         local_score[key] = scr;
         for(int k = ls_score + key; k < keis; k += ls_score)
            local_score[key] += score[shift_score + k];
        }
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      int count = ls_score;
      do
        {
         count = (count + 1) / 2;
         if(key < count)
           {
            if((key + count) < keis)
              {
               local_score[key] += local_score[key + count];
               local_score[key + count] = 0;
              }
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      score[shift_score] /= local_score[0];
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      shift_score -= key;
      for(int d = key; d < dimension; d += keis)
        {
         float sum = 0;
         int shift_value = (3 * variable * heads + 2 * heads + head) * dimension + d;
         for(int v = 0; v < keis; v++)
            sum += qkv[shift_value + v * shift] * score[shift_score + v];
         out[shift_out + d] = sum;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void HiddenGradientContAtt(__global float *qkv, __global float *qkv_g,
                                    __global float *dqkv,
                                    __global float *dqkv_g,
                                    __global float *score,
                                    __global float *out_g, const int dimension)
  {
   const size_t pos = get_global_id(0);
   const size_t variable = get_global_id(1);
   const size_t head = get_global_id(2);
   const size_t total = get_global_size(0);
   const size_t variables = get_global_size(1);
   const size_t heads = get_global_size(2);
//--- Value gradient
     {
      const int shift_value =
         dimension * (heads * (3 * variables * pos + 3 * variable + 2) + head);
      const int shift_out = dimension * (head + variable * heads);
      const int shift_score = total * (variable * heads + head);
      const int step_out = variables * heads * dimension;
      const int step_score = variables * heads * total;
      //---
      for(int d = 0; d < dimension; d++)
        {
         float sum = 0;
         for(int g = 0; g < total; g++)
            sum += out_g[shift_out + g * step_out + d] *
                   score[shift_score + g * step_score];
         qkv_g[shift_value + d] = sum;
        }
     }
//--- Query gradient
     {
      const int shift_out =
         dimension * (heads * (pos * variables + variable) + head);
      const int step = 3 * variables * heads * dimension;
      const int shift_query =
         dimension * (3 * heads * variable + head) + pos * step;
      const int shift_key = dimension * (heads * (3 * variable + 1) + head);
      const int shift_value = dimension * (heads * (3 * variable + 2) + head);
      const int shift_score =
         total * (heads * (pos * variables + variable) + head);
      //--- Score gradient
      for(int k = 0; k < total; k++)
        {
         float score_grad = 0;
         float scr = score[shift_score + k];
         for(int v = 0; v < total; v++)
           {
            float grad = 0;
            for(int d = 0; d < dimension; d++)
               grad += qkv[shift_value + v * step + d] * out_g[shift_out + d];
            score_grad += score[shift_score + v] * grad * ((float)(pos == v) - scr);
           }
         score_grad /= sqrt((float)dimension);
         //--- Query gradient
         for(int d = 0; d < dimension; d++)
           {
            if(k == 0)
              {
               dqkv_g[shift_query + d] = score_grad * qkv[shift_key + k * step + d];
               qkv_g[shift_query + d] = score_grad * dqkv[shift_key + k * step + d];
              }
            else
              {
               dqkv_g[shift_query + d] += score_grad * qkv[shift_key + k * step + d];
               qkv_g[shift_query + d] += score_grad * dqkv[shift_key + k * step + d];
              }
           }
        }
     }
//--- Key gradient
     {
      const int shift_key =
         dimension * (heads * (3 * variables * pos + 3 * variable + 1) + head);
      const int shift_out = dimension * (head + variable * heads);
      const int step_out = variables * heads * dimension;
      const int step = 3 * variables * heads * dimension;
      const int shift_query = dimension * (3 * heads * variable + head);
      const int shift_value =
         dimension * (heads * (3 * variable + 2) + head) + pos * step;
      const int shift_score = total * (heads * variable + head);
      const int step_score = variables * heads * total;
      //--- Score gradient
      for(int q = 0; q < total; q++)
        {
         float score_grad = 0;
         float scr = score[shift_score + q * step_score];
         for(int g = 0; g < total; g++)
           {
            float grad = 0;
            for(int d = 0; d < dimension; d++)
               grad += qkv[shift_value + d] * out_g[shift_out + d + g * step_out] / sqrt((float)dimension);
            score_grad += score[shift_score + q * step_score + g] * grad * ((float)(q == pos) - scr);
           }
         //--- Key gradient
         for(int d = 0; d < dimension; d++)
           {
            if(q == 0)
              {
               dqkv_g[shift_key + d] = qkv[shift_query + q * step + d] * score_grad;
               qkv_g[shift_key + d] = score_grad * dqkv[shift_query + q * step + d];
              }
            else
              {
               qkv_g[shift_key + d] += score_grad * dqkv[shift_query + q * step + d];
               dqkv_g[shift_key + d] += score_grad * qkv[shift_query + q * step + d];
              }
           }
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void RevInFeedForward(__global float *inputs, __global float *options,
                               __global float *output, int options_size,
                               int optimization)
  {
   int n = get_global_id(0);
   int shift = (n * (optimization == 0 ? 7 : 9)) % options_size;
//---
   float mean = options[shift];
   float variance = options[shift + 1];
   float k = options[shift + 3];
//---
   float res = 0;
   res = sqrt(variance) * (inputs[n] - options[shift + 4]) / fmax(k, 0.001f) + mean;
   if(isnan(res))
      res = 0;
//---
   output[n] = res;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void RevInHiddenGraddient(__global float *inputs, __global float *inputs_gr,
                                   __global float *options, __global float *output_gr,
                                   int options_size,
                                   int optimization,
                                   int activation)
  {
   int n = get_global_id(0);
   int shift = (n * (optimization == 0 ? 7 : 9)) % options_size;
//---
   float variance = options[shift + 1];
   float inp = inputs[n];
   float k = options[shift + 3];
//---
   float res = sqrt(variance) * output_gr[n];
   if(fabs(k) > 1)
      res /= k;
   if(isnan(res))
      res = 0;
//---
   inputs_gr[n] = Deactivation(res, inp, activation);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void DeActivation(__global const float *inputs, __global float *inputs_gr,
                           __global const float *output_gr, const int activation)
  {
   int n = get_global_id(0);
//---
   float inp = inputs[n];
   float res = output_gr[n];
   if(isnan(res))
      res = 0;
//---
   inputs_gr[n] = Deactivation(res, inp, activation);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void PatchCreate(__global float *inputs,
                          __global float *weights,
                          __global float *outputs,
                          int inputs_total,
                          int window_in,
                          int step,
                          int activation
                         )
  {
   const int i = get_global_id(0);
   const int w = get_global_id(1);
   const int v = get_global_id(2);
   const int window_out = get_global_size(1);
   const int variables = get_global_size(2);
//---
   const int shift_in = i * step * variables + v;
   const int shift_out = (i * variables + v) * window_out + w;
   const int shift_weights = (window_in + 1) * (v * window_out + w);
//---
   float res = weights[shift_weights + window_in];
   for(int p = 0; p < window_in; p++)
      if((shift_in + p * variables) < inputs_total)
         res += inputs[shift_in + p * variables] * weights[shift_weights + p];
   if(isnan(res))
      res = 0;
//---
   outputs[shift_out] = Activation(res, activation);;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void PatchHiddenGradient(__global float *inputs,
                                  __global float *inputs_gr,
                                  __global float *weights,
                                  __global float *outputs_gr,
                                  int window_in,
                                  int step,
                                  int window_out,
                                  int outputs_total,
                                  int activation
                                 )
  {
   const int i = get_global_id(0);
   const int v = get_global_id(1);
   const int variables = get_global_size(1);
//---
   const int w_start = i % step;
   const int r_start = max((i - window_in + step) / step, 0);
   int total = (window_in - w_start + step - 1) / step;
   total = min((i + step) / step, total);
//---
   float grad = 0;
   for(int p = 0; p < total; p ++)
     {
      int row = r_start + p;
      if(row >= outputs_total)
         break;
      for(int wo = 0; wo < window_out; wo++)
        {
         int shift_g = (row * variables + v) * window_out + wo;
         int shift_w = v * (window_in + 1) * window_out + w_start + (total - p - 1) * step + wo * (window_in + 1);
         grad += outputs_gr[shift_g] * weights[shift_w];
        }
     }
//---
   float inp = inputs[i * variables + v];
//---
   inputs_gr[i * variables + v] = Deactivation(grad, inp, activation);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void PatchUpdateWeightsAdam(__global float *weights,
                                     __global const float *outputs_gr,
                                     __global const float *inputs,
                                     __global float *weights_m,
                                     __global float *weights_v,
                                     const int inputs_total,
                                     const float l,
                                     const float b1,
                                     const float b2,
                                     int step
                                    )
  {
   const int c = get_global_id(0);
   const int r = get_global_id(1);
   const int v = get_global_id(2);
   const int window_in = get_global_size(0) - 1;
   const int window_out = get_global_size(1);
   const int variables = get_global_size(2);
//---
   const int start_input = c * variables + v;
   const int step_input = step * variables;
   const int start_out = v * window_out + r;
   const int step_out = variables * window_out;
   const int total = inputs_total / (variables * step);
//---
   float grad = 0;
   for(int p = 0; p < total; p++)
     {
      int i = start_input + i * step_input;
      int o = start_out + i * step_out;
      grad += (c == window_in ? 1 : inputs[i]) * outputs_gr[0];
     }
   if(isnan(grad))
      grad = 0;
//---
   const int shift_weights = (window_in + 1) * (window_out * v + r) + c;
//---
   float weight = weights[shift_weights];
   float mt = b1 * weights_m[shift_weights] + (1 - b1) * grad;
   float vt = b2 * weights_v[shift_weights] + (1 - b2) * pow(grad, 2);
   float delta = l * (mt / (sqrt(vt) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(fabs(delta) > 0)
      weights[shift_weights] = clamp(weight + delta, -MAX_WEIGHT, MAX_WEIGHT);
   weights_m[shift_weights] = mt;
   weights_v[shift_weights] = vt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MatMult(__global const float *matr1,
                      __global const float *matr2,
                      __global float *result,
                      int dimension)
  {
   size_t row = get_global_id(0);
   size_t col = get_global_id(1);
   size_t var = get_global_id(2);
   size_t rows = get_global_size(0);
   size_t cols = get_global_size(1);
//---
   int shift1 = (row  + var * rows) * dimension;
   int shift2 = col + var * dimension * cols;
   int shift_out = (row + var * rows) * cols + col;
//---
   float res = 0;
   for(int i = 0; i < dimension; i++)
      res += matr1[shift1 + i] * matr2[shift2 + i * cols];
//---
   result[shift_out] = res;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MatMultGrad(__global const float *matr1,
                          __global float *matr1_gr,
                          __global const float *matr2,
                          __global float *matr2_gr,
                          __global const float *result_gr,
                          int dimension)
  {
   size_t row = get_global_id(0);
   size_t col = get_global_id(1);
   size_t var = get_global_id(2);
   size_t rows = get_global_size(0);
   size_t cols = get_global_size(1);
//---
   int shift1 = (row  + var * rows) * dimension;
   int shift2 = var * dimension * cols;
   int shift_out = (row + var * rows) * cols;
//---
   for(int c = 0; c < dimension; c += cols)
     {
      if((c + col) >= dimension)
         continue;
      float grad = 0;
      for(int i = 0; i < cols; i++)
         grad += result_gr[shift_out + i] * matr2[shift2 + c * cols + i];
      matr1_gr[shift1 + c] = grad;
     }
//---
   shift_out = var * rows * cols + col;
   for(int r = 0; r < dimension; r += rows)
     {
      if((r + row) >= dimension)
         continue;
      shift1 = var * rows * dimension + r;
      float grad = 0;
      for(int i = 0; i < rows; i++)
         grad += result_gr[shift_out + i * cols] * matr1[shift1 + i * dimension];
      matr2_gr[shift2 + col + r * cols] = grad;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FFT(__global float *inputs_re,
                  __global float *inputs_im,
                  __global float *outputs_re,
                  __global float *outputs_im,
                  const int input_window,
                  const int input_complex,
                  const int output_window,
                  const int reverse
                 )
  {
   size_t variable = get_global_id(0);
//---
   const ulong N = output_window;
   const ulong N2 = N / 2;
   const ulong inp_shift = input_window * variable;
   const ulong out_shift = output_window * variable;
//---
   uint target = 0;
   for(uint position = 0; position < N; position++)
     {
      if(target > position)
        {
         outputs_re[out_shift + position] = (target < input_window ? inputs_re[inp_shift + target] : 0);
         outputs_im[out_shift + position] = ((target < input_window && input_complex) ? inputs_im[inp_shift + target] : 0);
         outputs_re[out_shift + target] = inputs_re[inp_shift + position];
         outputs_im[out_shift + target] = (input_complex ? inputs_im[inp_shift + position] : 0);
        }
      else
        {
         outputs_re[out_shift + position] = inputs_re[inp_shift + position];
         outputs_im[out_shift + position] = (input_complex ? inputs_im[inp_shift + position] : 0);
        }
      unsigned int mask = N;
      while(target & (mask >>= 1))
         target &= ~mask;
      target |= mask;
     }
   float real = 0, imag = 0;
   for(int len = 2; len <= (int)N; len <<= 1) //Строим итеративно БПФ для отрезков длины 2, 4, 8, ... n
     {
      float w_real = (float)cos(2 * M_PI_F / len);
      float w_imag = (float)sin(2 * M_PI_F / len); //Множитель, умножив на который, мы прокрутим аргумент на один пункт длины 2*pi/len дальше
      for(int i = 0; i < (int)N; i += len) //Обрабатываем поблочно, i - начало очередного блока
        {
         float cur_w_real = 1;
         float cur_w_imag = 0; //Аргумент
         for(int j = 0; j < len / 2; j++) //Итерируемся по блоку
           {
            real = cur_w_real * outputs_re[out_shift + i + j + len / 2] - cur_w_imag * outputs_im[out_shift + i + j + len / 2];
            imag = cur_w_imag * outputs_re[out_shift + i + j + len / 2] + cur_w_real * outputs_im[out_shift + i + j + len / 2];
            outputs_re[out_shift + i + j + len / 2] = outputs_re[out_shift + i + j] - real;
            outputs_im[out_shift + i + j + len / 2] = outputs_im[out_shift + i + j] - imag;
            outputs_re[out_shift + i + j] += real;
            outputs_im[out_shift + i + j] += imag;
            real = cur_w_real * w_real - cur_w_imag * w_imag; //Прокручиваем аргумент на угол 2*pi/len
            cur_w_imag = cur_w_imag * w_real + cur_w_real * w_imag; //Прокручиваем аргумент на угол 2*pi/len
            cur_w_real = real;
           } //Когда мы вышли из цикла, аргумент cur_w снова равен нулю, так как мы добавили 2*pi/len len раз.
        }
     }
//---
   if(reverse)
     {
      outputs_re[0] /= N;
      outputs_im[0] /= N;
      outputs_re[N2] /= N;
      outputs_im[N2] /= N;
      for(int i = 1; i < N2; i++)
        {
         real = outputs_re[i] / N;
         imag = outputs_im[i] / N;
         outputs_re[i] = outputs_re[N - i] / N;
         outputs_im[i] = outputs_im[N - i] / N;
         outputs_re[N - i] = real;
         outputs_im[N - i] = imag;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexLayer(__global float *inputs_re,
                           __global float *inputs_im,
                           __global float *outputs_re,
                           __global float *outputs_im
                          )
  {
   size_t i = get_global_id(0);
   size_t j = get_global_id(1);
   size_t total_i = get_global_size(0);
   size_t total_j = get_global_size(1);
   uint shift = i * total_j + j;
//---
   outputs_re[shift] = inputs_re[shift] - inputs_im[shift];
   outputs_im[shift] = inputs_im[shift] + inputs_re[shift];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexLayerGradient(__global float *inputs_re,
                                   __global float *inputs_im,
                                   __global float *outputs_re,
                                   __global float *outputs_im
                                  )
  {
   size_t i = get_global_id(0);
   size_t j = get_global_id(1);
   size_t total_i = get_global_size(0);
   size_t total_j = get_global_size(1);
   uint shift = i * total_j + j;
//---
   inputs_re[shift] = outputs_re[shift] + outputs_im[shift];
   inputs_im[shift] = outputs_im[shift] - outputs_re[shift];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void GradientMSA(__global float *matrix_t,  ///<[in] Target tensor
                          __global float *matrix_o,  ///<[in] Forecast tensor
                          __global float *matrix_g ///<[out] Tensor of gradients
                         )
  {
   int i = get_global_id(0);
   matrix_g[i] = matrix_t[i] - matrix_o[i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CumulativeGradient(__global float *gradient1,
                                 __global float *gradient2,
                                 __global float *gradient_out,
                                 float alpha
                                )
  {
   int i = get_global_id(0);
   gradient_out[i] = alpha * gradient1[i] + (1 - alpha) * gradient2[i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float2 ComplexMul(const float2 a, const float2 b)
  {
   float2 result = 0;
   result.x = a.x * b.x - a.y * b.y;
   result.y = a.x * b.y + a.y * b.x;
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float2 ComplexDiv(const float2 a, const float2 b)
  {
   float2 result = 0;
   float z = pow(b.x, 2) + pow(b.y, 2);
   if(z > 0)
     {
      result.x = (a.x * b.x + a.y * b.y) / z;
      result.y = (a.y * b.x - a.x * b.y) / z;
     }
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float ComplexAbs(float2 a)
  {
   return sqrt(pow(a.x, 2) + pow(a.y, 2));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float2 ComplexSqrt(float2 a)
  {
   float2 result = 0;
   float z = ComplexAbs(a);
   result.x = sqrt((z + a.x) / 2);
   result.y = sqrt((z - a.x) / 2);
   if(a.y < 0)
      result.y *= (-1);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float2 ComplexExp(float2 a)
  {
   float2 result = exp(clamp(a.x, -20.0f, 20.0f));
   result.x *= cos(a.y);
   result.y *= sin(a.y);
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float2 ComplexTanh(float2 a)
  {
   float sinh_re = sinh(a.x);
   float cosh_re = cosh(a.x);
   float sin_im = sin(a.y);
   float cos_im = cos(a.y);
//---
   float2 sinh_a = 0;
   float2 cosh_a = 0;
   sinh_a.x = sinh_re * cos_im;
   sinh_a.y = cosh_re * sin_im;
   cosh_a.x = cosh_re * cos_im;
   cosh_a.y = sinh_re * sin_im;
//---
   return ComplexDiv(sinh_a, cosh_a);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForwardComplexConv(__global float2 *matrix_w, ///<[in] Weights matrix (m+1)*n, where m - input
                                     ///< window and n - output window
                                     __global float2 *matrix_i, ///<[in] Inputs tensor
                                     __global float2 *matrix_o, ///<[out] Output tensor
                                     int inputs,               ///< Number of inputs
                                     int step,                 ///< Step size
                                     int window_in,            ///< Size of input window
                                     int activation            ///< Activation type (#ENUM_ACTIVATION)
                                    )
  {
   size_t i = get_global_id(0);
   size_t out = get_global_id(1);
   size_t w_out = get_global_size(1);
//---
   int w_in = window_in;
   int shift_out = w_out * i;
   int shift_in = step * i;
   int shift = (w_in + 1) * out;
   int stop = (w_in <= (inputs - shift_in) ? w_in : (inputs - shift_in));
//---
   float2 sum = ComplexMul((float2)(1, 0), matrix_w[shift + w_in]);
   for(int k = 0; k <= stop; k ++)
      sum += ComplexMul(matrix_i[shift_in + k], matrix_w[shift + k]);
   if(isnan(sum.x) || isnan(sum.y) || isinf(sum.x) || isinf(sum.y))
      sum = (float2)0;
//---
   switch(activation)
     {
      case 0:
         sum = ComplexTanh(sum);
         break;
      case 1:
         sum = ComplexDiv((float2)(1, 0), (float2)(1, 0) + ComplexExp(-sum));
         break;
      case 2:
         if(sum.x < 0)
            sum.x *= 0.01f;
         if(sum.y < 0)
            sum.y *= 0.01f;
         break;
      default:
         break;
     }
   matrix_o[out + shift_out] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CalcHiddenGradientComplexConv(__global float2 *matrix_w,     ///<[in] Weights matrix (m+1)*n, where m - input
      ///< window and n - output window
      __global float2 *matrix_g,  ///<[in] Tensor of gradients at current layer
      __global float2 *matrix_o,  ///<[in] Output tensor
      __global float2 *matrix_ig, ///<[out] Tensor of gradients at previous layer
      int outputs,               ///< Number of outputs
      int step,                  ///< Step size
      int window_in,             ///< Size of input window
      int window_out,            ///< Size of output window
      int activation,            ///< Activation type (#ENUM_ACTIVATION)
      int shift_out              ///< Shift in output and gradient buffer
                                           )
  {
   size_t i = get_global_id(0);
   size_t inputs = get_global_size(0);
//---
   float2 sum = (float2)0;
   float2 out = matrix_o[i];
   int start = i - window_in + step;
   start = max((start - start % step) / step, 0);
   int stop = (i + step - 1) / step;
   if(stop > (outputs / window_out))
      stop = outputs / window_out;
   for(int h = 0; h < window_out; h ++)
     {
      for(int k = start; k < stop; k++)
        {
         int shift_g = k * window_out + h;
         int shift_w = (stop - k - 1) * step + i % step + h * (window_in + 1);
         if(shift_g >= outputs || shift_w >= (window_in + 1) * window_out)
            break;
         sum += ComplexMul(matrix_g[shift_out + shift_g], matrix_w[shift_w]);
        }
     }
   if(isnan(sum.x) || isnan(sum.y) || isinf(sum.x) || isinf(sum.y))
      sum = (float2)0;
//---
   switch(activation)
     {
      case 0:
         sum = ComplexMul(sum, (float2)1.0f - ComplexMul(out, out));
         break;
      case 1:
         sum = ComplexMul(sum, ComplexMul(out, (float2)1.0f - out));
         break;
      case 2:
         if(out.x < 0.0f)
            sum.x *= 0.01f;
         if(out.y < 0.0f)
            sum.y *= 0.01f;
         break;
      default:
         break;
     }
   matrix_ig[i] = sum;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsComplexConvMomentum(__global float2 *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
      ///< input window and n - output window
      __global float2 *matrix_g, ///<[in] Tensor of gradients at current layer
      __global float2 *matrix_i, ///<[in] Inputs tensor
      __global float2 *matrix_dw, ///<[in,out] Matrix of delta weights in last correction
      int inputs,     ///< Number of inputs
      float learning_rates, ///< Learning rates
      float momentum,       ///< Momentum multiplier
      int window_in,        ///< Size of input window
      int window_out,       ///< Size of output window
      int step              ///< Step size
                                              )
  {
   const int i = get_global_id(0);
   const int shift = i % (window_in + 1);
   const int shift_out = (i - shift) / (window_in + 1);
   int total = (inputs - window_in) % step;
   total = (inputs - window_in - total) / step + (total > 0 ? 1 : 0);
   float2 grad = 0;
   for(int t = 0; t < total; t++)
     {
      if(shift != window_in && (shift + t * window_in) >= inputs)
         break;
      grad += ComplexMul(matrix_g[t * window_out + shift_out],
                         (shift == window_in ? (float2)(1, 0) : matrix_i[shift + t * step]));
     }
   float2 delta = ComplexMul((float2)(learning_rates, 0), grad) + ComplexMul((float2)(momentum, 0), matrix_dw[i]);
   if(!(isnan(delta.x) || isnan(delta.y) || isinf(delta.x) || isinf(delta.y)))
     {
      matrix_dw[i] = delta;
      matrix_w[i] += delta;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsComplexConvAdam(__global float2 *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
      ///< input window and n - output window
      __global const float2 *matrix_g, ///<[in] Tensor of gradients at current layer
      __global const float2 *matrix_i, ///<[in] Inputs tensor
      __global float2 *matrix_m,       ///<[in] Matrix of first momentum
      __global float2 *matrix_v,       ///<[in] Matrix of seconfd momentum
      const int inputs,               ///< Number of inputs
      const float l,                  ///< Learning rates
      const float b1,                 ///< First momentum multiplier
      const float b2,                 ///< Second momentum multiplier
      int window_in,                  ///< Size of input window
      int window_out,                 ///< Size of output window
      int step                        ///< Step size
                                          )
  {
   const int i = get_global_id(0);
   if(i > window_in)
      return;
//---
   int total = (inputs - (window_in - step)) % step;
   total = (inputs - (window_in - step) - total) / step + (total > 0 ? 1 : 0);
   for(int out = 0; out < window_out; out++)
     {
      float2 grad = 0;
      int shift_w = i + out * (window_in + 1);
      for(int t = 0; t < total; t++)
        {
         if(i != window_in && (i + t * window_in) >= inputs)
            break;
         grad += ComplexMul(matrix_g[t * window_out + out],
                            (i == window_in ? (float2)(1, 0) : matrix_i[i + t * step]));
        }
      float2 mt = ComplexMul((float2)(b1, 0), matrix_m[shift_w]) + ComplexMul((float2)(1 - b1, 0), grad);
      float2 vt = ComplexMul((float2)(b2, 0), matrix_v[shift_w]) + ComplexMul((float2)(1 - b2, 0), ComplexMul(grad, grad));
      float2 delta = ComplexDiv(ComplexMul((float2)(l, 0), mt), ComplexSqrt(vt));
      matrix_w[shift_w] = matrix_w[shift_w] + delta;
      matrix_m[shift_w] = mt;
      matrix_v[shift_w] = vt;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexSoftMax_FeedForward(__global float2 *inputs,
      __global float2 *outputs, const int total)
  {
   const uint i = (uint)get_global_id(0);
   const uint l = (uint)get_local_id(0);
   const uint h = (uint)get_global_id(1);
   const uint ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
   uint shift_head = h * total;
//---
   __local float2 temp[LOCAL_ARRAY_SIZE];
   uint count = 0;
   if(l < ls)
      do
        {
         uint shift = shift_head + count * ls + l;
         if(shift < ((h + 1) * total))
            temp[l].x = (count > 0 ? fmax(ComplexAbs(inputs[shift]), temp[l].x)
                         : ComplexAbs(inputs[shift]));
         count++;
        }
      while((count * ls + l) < total);
   barrier(CLK_LOCAL_MEM_FENCE);
   float max_value = temp[0].x;
   for(int i = 1; i < ls; i++)
      max_value = fmax(max_value, temp[i].x);
//---
   count = 0;
   if(l < ls)
      do
        {
         uint shift = shift_head + count * ls + l;
         temp[l] = (count > 0 ? temp[l] : (float2)0) +
                   (shift < ((h + 1) * total) ? ComplexExp(ComplexDiv(inputs[shift], (float2)(max_value, 0))) : (float2)0);
         count++;
        }
      while((count * ls + l) < total);
   barrier(CLK_LOCAL_MEM_FENCE);
   count = min(ls, (uint)total);
   do
     {
      count = (count + 1) / 2;
      if(l < ls)
         temp[l] += (l < count && (l + count) < total ? temp[l + count] : (float2)0);
      if(l + count < ls)
         temp[l + count] = (float2)0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   float2 sum = temp[0];
   if(ComplexAbs(sum) > 0)
     {
      count = 0;
      while((count * ls + l) < total)
        {
         uint shift = shift_head + count * ls + l;
         if(shift < ((h + 1) * total))
            outputs[shift] = ComplexDiv(ComplexExp(ComplexDiv(inputs[shift], (float2)(max_value, 0))), sum);
         count++;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexSoftMax_HiddenGradient(__global float2 *outputs,
      __global float2 *output_gr,
      __global float2 *input_gr)
  {
   size_t i = get_global_id(0);
   size_t outputs_total = get_global_size(0);
   size_t h = get_global_id(1);
   uint shift = h * outputs_total;
   float2 output = outputs[shift + i];
   float2 result = 0;
   for(int j = 0; j < outputs_total; j++)
      result += ComplexMul(ComplexMul(outputs[shift + j], output_gr[shift + j]), ((i == j ? (float2)(1, 0) : (float2)0) - output));
   input_gr[shift + i] = result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexSoftMax_OutputGradient(__global float2 *outputs,
      __global float2 *targets,
      __global float2 *output_gr)
  {
   size_t i = get_global_id(0);
   if(ComplexAbs(outputs[i]) == 0)
      output_gr[i] = (float2)0;
   else
      output_gr[i] = ComplexDiv(targets[i], outputs[i]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexMHAttentionScore(__global float2 *qkv,   ///<[in] Matrix of Querys, Keys, Values
                                      __global float2 *score, ///<[out] Matrix of Scores
                                      int dimension,         ///< Dimension of Key
                                      int mask ///< 1 - calc only previous units, 0 - calc all
                                     )
  {
   int q = get_global_id(0);
   int h = get_global_id(1);
   int units = get_global_size(0);
   int heads = get_global_size(1);
//---
   int shift_q = dimension * (h + 3 * q * heads);
   int shift_s = units * (h + q * heads);
//---
   float2 koef = (float2)(sqrt((float)dimension), 0);
   if(koef.x < 1)
      koef.x = 1;
   float2 sum = 0;
   for(int k = 0; k < units; k++)
     {
      if(mask > 0 && k > q)
        {
         score[shift_s + k] = (float2)0;
         continue;
        }
      float2 result = (float2)0;
      int shift_k = dimension * (h + heads * (3 * k + 1));
      for(int i = 0; i < dimension; i++)
         result += ComplexMul(qkv[shift_q + i], qkv[shift_k + i]);
      result = ComplexExp(ComplexDiv(result, koef));
      if(isnan(result.x) || isnan(result.y) || isinf(result.x) || isinf(result.y))
         result = (float2)0;
      score[shift_s + k] = result;
      sum += result;
     }
   if(ComplexAbs(sum) > 0)
      for(int k = 0; k < units; k++)
         score[shift_s + k] = ComplexDiv(score[shift_s + k], sum);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexMHAttentionOut(__global float2 *scores, ///<[in] Matrix of Scores
                                    __global float2 *qkv,    ///<[in] Matrix of Values
                                    __global float2 *out,    ///<[out] Output tensor
                                    int dimension           ///< Dimension of Value
                                   )
  {
   int u = get_global_id(0);
   int units = get_global_size(0);
   int h = get_global_id(1);
   int heads = get_global_size(1);
//---
   int shift_s = units * (h + heads * u);
   int shift_out = dimension * (h + heads * u);
//---
   for(int d = 0; d < dimension; d++)
     {
      float2 result = (float2)0;
      for(int v = 0; v < units; v++)
        {
         int shift_v = dimension * (h + heads * (3 * v + 2)) + d;
         result += ComplexMul(scores[shift_s + v], qkv[shift_v]);
        }
      out[shift_out + d] = result;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexMHAttentionGradients(__global float2 *qkv, __global float2 *qkv_g,
      __global float2 *scores, __global float2 *gradient)
  {
   size_t u = get_global_id(0);
   size_t h = get_global_id(1);
   size_t d = get_global_id(2);
   size_t units = get_global_size(0);
   size_t heads = get_global_size(1);
   size_t dimension = get_global_size(2);
//---
   float2 koef = (float2)(sqrt((float)dimension), 0);
   if(koef.x < 1)
      koef.x = 1;
//--- init
   const int shift_q = dimension * (heads * 3 * u + h);
   const int shift_k = dimension * (heads * (3 * u + 1) + h);
   const int shift_v = dimension * (heads * (3 * u + 2) + h);
   const int shift_g = dimension * (heads * u + h);
   int shift_score = h * units;
   int step_score = units * heads;
//--- Calculating Value's gradients
   float2 sum = (float2)0;
   for(int i = 0; i < units; i++)
      sum += ComplexMul(gradient[(h + i * heads) * dimension + d], scores[shift_score + u + i * step_score]);
   qkv_g[shift_v + d] = sum;
//--- Calculating Query's gradients
   shift_score = h * units + u * step_score;
   float2 grad = 0;
   float2 grad_out = gradient[shift_g + d];
   for(int k = 0; k < units; k++)
     {
      float2 sc_g = (float2)0;
      float2 sc = scores[shift_score + k];
      for(int v = 0; v < units; v++)
         sc_g += ComplexMul(
                    ComplexMul(scores[shift_score + v],
                               ComplexMul(qkv[dimension * (heads * (3 * v + 2) + h)],
                                          grad_out)),
                    ((float2)(k == v, 0) - sc)
                 );
      grad += ComplexMul(ComplexDiv(sc_g, koef), qkv[dimension * (heads * (3 * k + 1) + h) + d]);
     }
   qkv_g[shift_q + d] = grad;
//--- Calculating Key's gradients
   grad = 0;
   for(int q = 0; q < units; q++)
     {
      shift_score = h * units + q * step_score;
      float2 sc_g = (float2)0;
      float2 sc = scores[shift_score + u];
      float2 grad_out = gradient[dimension * (heads * q + h) + d];
      for(int v = 0; v < units; v++)
         sc_g += ComplexMul(
                    ComplexMul(scores[shift_score + v],
                               ComplexMul(qkv[dimension * (heads * (3 * v + 2) + h)],
                                          grad_out)),
                    ((float2)(u == v, 0) - sc)
                 );
      grad += ComplexMul(ComplexDiv(sc_g, koef), qkv[dimension * (heads * 3 * q + h) + d]);
     }
   qkv_g[shift_k + d] = grad;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexNormalize(__global float2 *inputs,
                               __global float2 *outputs,
                               __global float2 *means,
                               __global float *vars,
                               int dimension)
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
   const float2 dim = (float2)(dimension, 0);
//---
   float2 mean = 0;
   for(int i = 0; i < dimension; i++)
     {
      float2 val = inputs[shift + i];
      if(isnan(val.x) || isinf(val.x) ||
         isnan(val.y) || isinf(val.y))
         inputs[shift + i] = (float2)0;
      else
         mean += val;
     }
   means[n] = mean = ComplexDiv(mean, dim);
   float variance = 0;
   for(int i = 0; i < dimension; i++)
      variance += pow(ComplexAbs(inputs[shift + i] - mean), 2);
   vars[n] = variance = sqrt((isnan(variance) || isinf(variance) ? 1.0f : variance / dimension));
   float2 v = (float2)(variance, 0);
   for(int i = 0; i < dimension; i++)
     {
      float2 val = ComplexDiv((inputs[shift + i] - mean), v);
      if(isnan(val.x) || isinf(val.x) || isnan(val.y) || isinf(val.y))
         val = (float2)0;
      outputs[shift + i] = val;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexNormalizeGradient(__global float2 *inputs_gr,
                                       __global float2 *outputs_gr,
                                       __global float *vars,
                                       int dimension)
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float v = vars[n];
   float2 variance = (float2)((v > 0 ? v : 1.0f), 0);
   for(int i = 0; i < dimension; i++)
     {
      float2 val = ComplexDiv(outputs_gr[shift + i], variance);
      if(isnan(val.x) || isinf(val.x) || isnan(val.y) || isinf(val.y))
         val = (float2)0;
      inputs_gr[shift + i] = val;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexUnNormalize(__global float2 *inputs,
                                 __global float2 *outputs,
                                 __global float2 *means,
                                 __global float *vars,
                                 int dimension)
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float v = vars[n];
   float2 variance = (float2)((v > 0 ? v : 1.0f), 0);
   float2 mean = means[n];
   for(int i = 0; i < dimension; i++)
     {
      float2 val = ComplexMul(inputs[shift + i], variance) + mean;
      if(isnan(val.x) || isinf(val.x) || isnan(val.y) || isinf(val.y))
         val = (float2)0;
      outputs[shift + i] = val;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void ComplexUnNormalizeGradient(__global float2 *inputs_gr,
      __global float2 *outputs_gr,
      __global float *vars,
      int dimension)
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float v = vars[n];
   float2 variance = (float2)((v > 0 ? v : 1.0f), 0);
   for(int i = 0; i < dimension; i++)
     {
      float2 val = ComplexMul(outputs_gr[shift + i], variance);
      if(isnan(val.x) || isinf(val.x) || isnan(val.y) || isinf(val.y))
         val = (float2)0;
      inputs_gr[shift + i] = val;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MainFreqWeight(__global float2 *freq,
                             __global float *weight,
                             int dimension
                            )
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float max_f = 0;
   float total = 0;
   float energy;
   for(int i = 0; i < dimension; i++)
     {
      energy = ComplexAbs(freq[shift + i]);
      total += energy;
      max_f = fmax(max_f, energy);
     }
   weight[n] = max_f / (total > 0 ? total : 1);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void WeightedSum(__global float *inputs1,
                          __global float *inputs2,
                          __global float *outputs,
                          __global float *weight,
                          int dimension
                         )
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float w = weight[n];
   for(int i = 0; i < dimension; i++)
      outputs[shift + i] = inputs1[shift + i] * w + inputs2[shift + i] * (1 - w);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void WeightedSumGradient(__global float *inputs_gr1,
                                  __global float *inputs_gr2,
                                  __global float *outputs_gr,
                                  __global float *weight,
                                  int dimension
                                 )
  {
   if(dimension <= 0)
      return;
//---
   size_t n = get_global_id(0);
   const int shift = n * dimension;
//---
   float w = weight[n];
   float w1 = 1 - weight[n];
   for(int i = 0; i < dimension; i++)
     {
      float grad = outputs_gr[shift + i];
      inputs_gr1[shift + i] = grad * w;
      inputs_gr2[shift + i] = grad * w1;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForwardS3(__global float* inputs,
                            __global float* probability,
                            __global float* weights,
                            __global float* outputs,
                            __global float* positions,
                            const int window,
                            const int total
                           )
  {
   int pos = get_global_id(0);
   int segments = get_global_size(0);
//---
   if((segments * window) > total)
      segments--;
//---
   int segment = 0;
   if(pos < segments)
     {
      const float prob = probability[pos];
      for(int i = 0; i < pos; i++)
        {
         if(probability[i] <= prob)
            segment++;
        }
      for(int i = pos + 1; i < segments; i++)
        {
         if(probability[i] < prob)
            segment++;
        }
     }
   else
      segment = pos;
//---
   const int shift_in = segment * window;
   const int shift_out = pos * window;
   const float w1 = weights[0];
   const float w2 = weights[1];
   positions[pos] = (float)segment;
   for(int i = 0; i < window; i++)
     {
      if((shift_in + i) >= total || (shift_out + i) >= total)
         break;
      outputs[shift_out + i] = w1 * inputs[shift_in + i] + w2 * inputs[shift_out + i];
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void InsideGradientS3(__global float* inputs,
                               __global float* inputs_gr,
                               __global float* probability,
                               __global float* probability_gr,
                               __global float* weights,
                               __global float* outputs_gr,
                               __global float* positions,
                               const int window,
                               const int total
                              )
  {
   size_t pos = get_global_id(0);
//---
   int segment = (int)positions[pos];
   float prob = probability[pos];
   const float w1 = weights[0];
   const float w2 = weights[1];
   const int shift_in = segment * window;
   const int shift_out = pos * window;
//---
   float grad = 0;
   float temp = 0;
   for(int i = 0; i < window; i++)
     {
      if((shift_out + i) >= total)
         break;
      temp = outputs_gr[shift_out + i] * w1;
      grad += temp * inputs[shift_in + i];
      inputs_gr[shift_in + i] = temp + outputs_gr[shift_in + i] * w2;
     }
   probability_gr[segment] = grad / prob;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void WeightGradientS3(__global float *inputs,
                               __global float *positions,
                               __global float *outputs_gr,
                               __global float *weights_gr,
                               const int window,
                               const int total
                              )
  {
   size_t l = get_local_id(0);
   size_t w = get_global_id(1);
   size_t ls = min((uint)get_local_size(0), (uint)LOCAL_ARRAY_SIZE);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
//---
   if(l < ls)
     {
      float val = 0;
      //---
      for(int i = l; i < total; i += ls)
        {
         int shift_in = i;
         if(w == 0)
           {
            int pos = i / window;
            shift_in = positions[pos] * window + i % window;
           }
         val += outputs_gr[i] * inputs[shift_in];
        }
      temp[l] = val;
     }
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   int t = ls;
   do
     {
      t = (t + 1) / 2;
      if(l < t && (l + t) < ls)
        {
         temp[l] += temp[l + t];
         temp[l + t] = 0;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(t > 1);
//---
   if(l == 0)
      weights_gr[w] = temp[0];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MH2PyrAttentionOut(__global float *q,
                                 __global float *kv,
                                 __global float *score,
                                 __global float *out,
                                 const int dimension,
                                 const int heads_kv,
                                 const int window
                                )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_k = dimension * (2 *  heads_kv * k + h_kv);
   const int shift_v = dimension * (2 *  heads_kv * k + heads_kv + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   const int delta_win = (window + 1) / 2;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- sum of exp
   uint count = 0;
   if(k < ls)
      do
        {
         if((count * ls) < (kunits - k))
           {
            float sum = 0;
            if(abs(count * ls + k - q_id) <= delta_win)
              {
               int sh_k = 2 * dimension * heads_kv * count * ls;
               for(int d = 0; d < dimension; d++)
                  sum = q[shift_q + d] * kv[shift_k + d + sh_k];
               sum = exp(sum / koef);
               if(isnan(sum))
                  sum = 0;
              }
            temp[k] = (count > 0 ? temp[k] : 0) + sum;
           }
         count++;
        }
      while((count * ls + k) < kunits);
   barrier(CLK_LOCAL_MEM_FENCE);
   count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k < ls)
         temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
      if(k + count < ls)
         temp[k + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- score
   float sum = temp[0];
   float sc = 0;
   if(sum != 0 && abs(k - q_id) <= delta_win)
     {
      for(int d = 0; d < dimension; d++)
         sc = q[shift_q + d] * kv[shift_k + d];
      sc = exp(sc / koef) / sum;
      if(isnan(sc))
         sc = 0;
     }
   score[shift_s] = sc;
   barrier(CLK_LOCAL_MEM_FENCE);
//--- out
   for(int d = 0; d < dimension; d++)
     {
      uint count = 0;
      if(k < ls)
         do
           {
            if((count * ls) < (kunits - k))
              {
               float sum = 0;
               if(abs(count * ls + k - q_id) <= delta_win)
                 {
                  int sh_v = 2 * dimension * heads_kv * count * ls;
                  sum = kv[shift_v + d + sh_v] * (count == 0 ? sc : score[shift_s + count * ls]);
                  if(isnan(sum))
                     sum = 0;
                 }
               temp[k] = (count > 0 ? temp[k] : 0) + sum;
              }
            count++;
           }
         while((count * ls + k) < kunits);
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k < ls)
            temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
         if(k + count < ls)
            temp[k + count] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      out[shift_q + d] = temp[0];
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void PLR(__global const float *inputs,
                  __global float *outputs,
                  __global int *isttp,
                  const int transpose,
                  const float min_step
                 )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = ((bool)transpose ? (i * variables + v) : (v * lenth + i));
   const int step_in = ((bool)transpose ? variables : 1);
//--- look for ttp
   float value = inputs[shift_in];
   bool bttp = false;
   if(i == 0 || i == lenth - 1)
      bttp = true;
   else
     {
      float prev = value;
      int prev_pos = i;
      float max_v = value;
      float max_pos = i;
      float min_v = value;
      float min_pos = i;
      while(fmax(fabs(prev - max_v), fabs(prev - min_v)) < min_step && prev_pos > 0)
        {
         prev_pos--;
         prev = inputs[shift_in - (i - prev_pos) * step_in];
         if(prev >= max_v && (prev - min_v) < min_step)
           {
            max_v = prev;
            max_pos = prev_pos;
           }
         if(prev <= min_v && (max_v - prev) < min_step)
           {
            min_v = prev;
            min_pos = prev_pos;
           }
        }
      //---
      float next = value;
      int next_pos = i;
      while(fmax(fabs(next - max_v), fabs(next - min_v)) < min_step && next_pos < (lenth - 1))
        {
         next_pos++;
         next = inputs[shift_in + (next_pos - i) * step_in];
         if(next > max_v && (next - min_v) < min_step)
           {
            max_v = next;
            max_pos = next_pos;
           }
         if(next < min_v && (max_v - next) < min_step)
           {
            min_v = next;
            min_pos = next_pos;
           }
        }
      //---
      if(
         (value >= prev && value > next) ||
         (value > prev && value == next) ||
         (value <= prev && value < next) ||
         (value < prev && value == next)
      )
         if(max_pos == i || min_pos == i)
            bttp = true;
     }
//---
   isttp[shift_in] = (int)bttp;
   outputs[shift_in] = 0;
   barrier(CLK_LOCAL_MEM_FENCE);
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int prev_ttp = 0;
   if(bttp)
     {
      pos = 0;
      for(int p = 0; p < i; p++)
        {
         int current_in = ((bool)transpose ? (p * variables + v) : (v * lenth + p));
         if((bool)isttp[current_in])
           {
            pos++;
            prev_ttp = p;
            prev_in = current_in;
           }
        }
     }
//--- cacl tendency
   if(pos > 0 && pos < (lenth / 3))
     {
      float sum_x = 0;
      float sum_y = 0;
      float sum_xy = 0;
      float sum_xx = 0;
      int dist = i - prev_ttp;
      for(int p = 0; p < dist; p++)
        {
         float x = (float)(p);
         float y = inputs[prev_in + p * step_in];
         sum_x += x;
         sum_y += y;
         sum_xy += x * y;
         sum_xx += x * x;
        }
      float slope = (dist * sum_xy - sum_x * sum_y) / (dist > 1 ? (dist * sum_xx - sum_x * sum_x) : 1);
      float intercept = (sum_y - slope * sum_x) / dist;
      int shift_out = ((bool)transpose ? ((pos - 1) * 3 * variables + v) : (v * lenth + (pos - 1) * 3));
      outputs[shift_out] = slope;
      outputs[shift_out + step_in] = intercept;
      outputs[shift_out + 2 * step_in] = ((float)dist) / lenth;
     }
   else
     {
      if(pos == (lenth / 3))
        {
         float sum_x = 0;
         float sum_y = 0;
         float sum_xy = 0;
         float sum_xx = 0;
         int dist = lenth - prev_ttp;
         for(int p = 0; p < dist; p++)
           {
            float x = (float)(p);
            float y = inputs[prev_in + p * step_in];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
           }
         float slope = (dist * sum_xy - sum_x * sum_y) / (dist > 1 ? (dist * sum_xx - sum_x * sum_x) : 1);
         float intercept = (sum_y - slope * sum_x) / dist;
         int shift_out = ((bool)transpose ? ((pos - 1) * 3 * variables + v) : (v * lenth + (pos - 1) * 3));
         outputs[shift_out] = slope;
         outputs[shift_out + step_in] = intercept;
         outputs[shift_out + 2 * step_in] = ((float)dist) / lenth;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void PLRGradient(__global float *inputs_gr,
                          __global const float *outputs,
                          __global const float *outputs_gr,
                          const int transpose
                         )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = ((bool)transpose ? (i * variables + v) : (v * lenth + i));
   const int step_in = ((bool)transpose ? variables : 1);
   const int shift_out = ((bool)transpose ? v : (v * lenth));
   const int step_out = 3 * step_in;
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int dist = 0;
   do
     {
      pos++;
      prev_in += dist;
      dist = (int)fmax(outputs[shift_out + pos * step_out + 2 * step_in] * lenth, 1);
     }
   while(!(prev_in <= i && (prev_in + dist) > i));
//--- calc constants
   float sum_x = 0;
   float sum_xx = 0;
   for(int p = 0; p < dist; p++)
     {
      float x = (float)(p);
      sum_x += x;
      sum_xx += x * x;
     }
//--- get output gradient
   float grad_slope = outputs_gr[shift_out + pos * step_out];
   float grad_intercept = outputs_gr[shift_out + pos * step_out + step_in];
//--- calc gradient
   grad_slope -= sum_x / dist * grad_intercept;
   grad_slope /= fmax(dist * sum_xx - sum_x * sum_x, 1);
   float grad = grad_intercept / dist;
   grad += (dist * (i - prev_in) - sum_x) * grad_slope;
   if(isnan(grad) || isinf(grad))
      grad = 0;
//--- save result
   inputs_gr[shift_in] = grad;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsAdamMini(__global float *matrix_w, ///<[in,out] Weights matrix (m+1)*n, where m -
                                    ///< number of neurons in previous layer and n -
                                    ///< number of neurons in current layer
                                    __global const float *matrix_g, ///<[in] Tensor of gradients at current layer
                                    __global const float *matrix_i, ///<[in] Inputs tensor
                                    __global float *matrix_m,       ///<[in,out] Matrix of first momentum
                                    __global float *matrix_v,       ///<[in,out] Matrix of seconfd momentum
                                    const float l,                  ///< Learning rates
                                    const float b1,                 ///< First momentum multiplier
                                    const float b2                  ///< Second momentum multiplier
                                   )
  {
//--- inputs
   const size_t i = get_local_id(0);
   const size_t inputs = get_local_size(0) - 1;
//--- outputs
   const size_t o = get_global_id(1);
   const size_t outputs = get_global_size(1);
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((uint)LOCAL_ARRAY_SIZE, (uint)inputs);
   const float inp = (i < inputs ? matrix_i[i] : 1.0f);
   int count = 0;
   do
     {
      if(count == (i / ls))
        {
         int shift = i % ls;
         temp[shift] = (count == 0 ? 0 : temp[shift]) + ((isnan(inp) || isinf(inp)) ? 0 : inp * inp) / inputs;
        }
      count++;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count * ls < inputs);
//--- sum
   count = (ls + 1) / 2;
   do
     {
      if(i < count && (i + count) < ls)
        {
         temp[i] += temp[i + count];
         temp[i + count] = 0;
        }
      count = (count + 1) / 2;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- calc v
   if(i == 0)
     {
      temp[1] = matrix_g[o];
      if(isnan(temp[1]) || isinf(temp[1]))
         temp[1] = 0;
      if(isnan(temp[0]) || isinf(temp[0]))
         temp[0] = 1;
      float v = matrix_v[o];
      if(isnan(v) || isinf(v))
         v = 1;
      temp[0] = b2 * v + (1 - b2) * pow(temp[1], 2) * temp[0];
      matrix_v[o] = temp[0];
     }
   barrier(CLK_LOCAL_MEM_FENCE);
//---
   const int wi = o * (inputs + 1) + i;
   float weight = matrix_w[wi];
   if(isnan(weight) || isinf(weight))
      weight = 0;
//---
   float m = matrix_m[wi];
   if(isnan(m) || isinf(m))
      m = 0;
//--- calc m
   m = b1 * m + (1 - b1) * temp[1] * inp;
   if(isnan(m) || isinf(m))
      m = 0;
//---
   float delta = l * (m / (sqrt(temp[0]) + 1.0e-37f) - (l1 * sign(weight) + l2 * weight));
   if(isnan(delta) || isinf(delta))
      delta = 0;
   if(delta > 0)
      matrix_w[wi] = clamp(weight + delta, -MAX_WEIGHT, MAX_WEIGHT);
   matrix_m[wi] = m;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UpdateWeightsConvAdamMini(__global float *matrix_w,    ///<[in,out] Weights matrix (m+1)*n, where m -
                                        ///< input window and n - output window
                                        __global const float *matrix_g, ///<[in] Tensor of gradients at current layer
                                        __global const float *matrix_i, ///<[in] Inputs tensor
                                        __global float *matrix_m,       ///<[in] Matrix of first momentum
                                        __global float *matrix_v,       ///<[in] Matrix of seconfd momentum
                                        const int inputs,               ///< Number of inputs
                                        const float l,                  ///< Learning rates
                                        const float b1,                 ///< First momentum multiplier
                                        const float b2,                 ///< Second momentum multiplier
                                        int step                        ///< Step size
                                       )
  {
//--- window in
   const size_t i = get_global_id(0);
   const size_t window_in = get_global_size(0) - 1;
//--- window out
   const size_t f = get_global_id(1);
   const size_t window_out = get_global_size(1);
//--- head window out
   const size_t f_h = get_local_id(1);
   const size_t window_out_h = get_local_size(1);
//--- variable
   const size_t v = get_global_id(2);
   const size_t variables = get_global_size(2);
//--- constants
   const int total = (inputs - window_in + step - 1) / step;
   const int shift_var_in = v * inputs;
   const int shift_var_out = v * total * window_out;
   const int shift_w = (f + v * window_out) * (window_in + 1) + i;
//---
   __local float temp[LOCAL_ARRAY_SIZE];
   const int ls = min((uint)window_in, (uint)LOCAL_ARRAY_SIZE);
//--- calc gradient
   float grad = 0;
   for(int t = 0; t < total; t++)
     {
      if(i != window_in && (i + t * window_in) >= inputs)
         break;
      float gt = matrix_g[t * window_out + f + shift_var_out] *
                 (i == window_in ? 1 : matrix_i[i + t * step + shift_var_in]);
      if(!(isnan(gt) || isinf(gt)))
         grad += gt;
     }
//--- calc sum grad
   int count;
   for(int h = 0; h < window_out_h; h++)
     {
      count = 0;
      do
        {
         if(h == f_h)
           {
            if(count == (i / ls))
              {
               int shift = i % ls;
               temp[shift] = ((count == 0 && h == 0) ? 0 : temp[shift]) + ((isnan(grad) || isinf(grad)) ? 0 : grad * grad) / (window_in * window_out_h);
              }
           }
         count++;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while((count * ls) < window_in);
     }
   count = (ls + 1) / 2;
   do
     {
      if(i < count && (i + count) < ls && f_h == 0)
        {
         temp[i] += temp[i + count];
         temp[i + count] = 0;
        }
      count = (count + 1) / 2;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- calc v
   if(i == 0 && f_h == 0)
     {
      if(isnan(temp[0]) || isinf(temp[0]))
         temp[0] = 1;
      int head = f / window_out_h;
      float v = matrix_v[head];
      if(isnan(v) || isinf(v))
         v = 1;
      temp[0] = clamp(b2 * v + (1 - b2) * temp[0], 1.0e-6f, 1.0e6f);
      matrix_v[head] = temp[0];
     }
   barrier(CLK_LOCAL_MEM_FENCE);
//--- calc m
   float mt = clamp(b1 * matrix_m[shift_w] + (1 - b1) * grad, -1.0e5f, 1.0e5f);
   if(isnan(mt) || isinf(mt))
      mt = 0;
   float weight = clamp(matrix_w[shift_w] + l * mt / sqrt(temp[0]), -MAX_WEIGHT, MAX_WEIGHT);
   if(!(isnan(weight) || isinf(weight)))
      matrix_w[shift_w] = weight;
   matrix_m[shift_w] = mt;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CutTrendAndOther(__global const float *inputs,
                               __global const float *plr,
                               __global float *trend,
                               __global float *other
                              )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = i * variables + v;
   const int step_in = variables;
   const int shift_plr = v;
   const int step_plr = 3 * step_in;
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int dist = 0;
   do
     {
      pos++;
      prev_in += dist;
      dist = (int)fmax(plr[shift_plr + pos * step_plr + 2 * step_in] * lenth, 1);
     }
   while(!(prev_in <= i && (prev_in + dist) > i));
//--- calc trend
   float sloat = plr[shift_plr + pos * step_plr];
   float intercept = plr[shift_plr + pos * step_plr + step_in];
   pos = i - prev_in;
   float trend_i = sloat * pos + intercept;
   float other_i = inputs[shift_in] - trend_i;
//--- save result
   trend[shift_in] = trend_i;
   other[shift_in] = other_i;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CutTrendAndOtherGradient(__global float *inputs_gr,
                                       __global const float *plr,
                                       __global float *plr_gr,
                                       __global const float *trend_gr,
                                       __global const float *other_gr
                                      )
  {
   const size_t i = get_global_id(0);
   const size_t lenth = get_global_size(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//--- constants
   const int shift_in = i * variables + v;
   const int step_in = variables;
   const int shift_plr = v;
   const int step_plr = 3 * step_in;
//--- calc position
   int pos = -1;
   int prev_in = 0;
   int dist = 0;
   do
     {
      pos++;
      prev_in += dist;
      dist = (int)fmax(plr[shift_plr + pos * step_plr + 2 * step_in] * lenth, 1);
     }
   while(!(prev_in <= i && (prev_in + dist) > i));
//--- get gradient
   float other_i_gr = other_gr[shift_in];
   float trend_i_gr = trend_gr[shift_in] - other_i_gr;
//--- calc plr gradient
   pos = i - prev_in;
   float sloat_gr = trend_i_gr * pos;
   float intercept_gr = trend_i_gr;
//--- save result
   plr_gr[shift_plr + pos * step_plr] += sloat_gr;
   plr_gr[shift_plr + pos * step_plr + step_in] += intercept_gr;
   inputs_gr[shift_in] = other_i_gr;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CutOneFromAnother(__global const float *inputs,
                                __global const float *cut,
                                __global float *other
                               )
  {
   const size_t i = get_global_id(0);
//--- save result
   other[i] = inputs[i] - cut[i];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CutOneFromAnotherGradient(__global float *inputs_gr,
                                        __global float *cut_gr,
                                        __global const float *other_gr
                                       )
  {
   const size_t i = get_global_id(0);
   float gr = other_gr[i];
//--- save result
   inputs_gr[i] = gr;
   cut_gr[i] = (-gr);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UniTrajPrepare(__global const float *history,
                             __global const float *h_mask,
                             __global const float *future,
                             __global const float *f_mask,
                             __global float *output,
                             const int h_total,
                             const int f_total
                            )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//---
   const int shift_in = i * variables + v;
   const int shift_out = 3 * shift_in;
   const int shift_f_out = 3 * (h_total * variables + v);
//--- history
   if(i < h_total)
     {
      float mask = h_mask[shift_in];
      float h = history[shift_in];
      float v = (i < (h_total - 1) && mask != 0 ? (history[shift_in + variables] - h) * mask : 0);
      if(isnan(v) || isinf(v))
         v = h = mask = 0;
      output[shift_out] = h * mask;
      output[shift_out + 1] = v;
      output[shift_out + 2] = mask;
     }
//--- future
   if(i < f_total)
     {
      float mask = f_mask[shift_in];
      float f = future[shift_in];
      float v = (i < (f_total - 1) && mask != 0 ? (future[shift_in + variables] - f) * mask : 0);
      if(isnan(v) || isinf(v))
         v = f = mask = 0;
      output[shift_f_out + shift_out] = f * mask;
      output[shift_f_out + shift_out + 1] = v;
      output[shift_f_out + shift_out + 2] = mask;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UniTrajPrepareGrad(__global float *history_gr,
                                 __global float *future_gr,
                                 __global const float *output,
                                 __global const float *output_gr,
                                 const int h_total,
                                 const int f_total
                                )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//---
   const int shift_in = i * variables + v;
   const int shift_out = 3 * shift_in;
   const int shift_f_out = 3 * (h_total * variables + v);
//--- history
   if(i < h_total)
     {
      float mask = output[shift_out + 2];
      float grad = 0;
      if(mask > 0)
        {
         grad = output_gr[shift_out] * mask;
         grad -= (i < (h_total - 1) && mask != 0 ? (output_gr[shift_out + 1]) * mask : 0);
         grad += (i > 0 ? output[shift_out + 1 - 3 * variables] * output[shift_out + 2 - 3 * variables] : 0);
         if(isnan(grad) || isinf(grad))
            grad = 0;
         //---
        }
      history_gr[shift_in] = grad;
     }
//--- future
   if(i < f_total)
     {
      float mask = output[shift_f_out + shift_out + 2];
      float grad = 0;
      if(mask > 0)
        {
         grad = output_gr[shift_f_out + shift_out] * mask;
         grad -= (i < (h_total - 1) && mask != 0 ? (output_gr[shift_f_out + shift_out + 1]) * mask : 0);
         grad += (i > 0 ? output[shift_f_out + shift_out + 1 - 3 * variables] * output[shift_f_out + shift_out + 2 - 3 * variables] : 0);
         if(isnan(grad) || isinf(grad))
            grad = 0;
         //---
        }
      future_gr[shift_in] = grad;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void UniTrajBTS(__global const float * concat_inp,
                         __global float * d_forw,
                         __global float * d_bakw,
                         const int total
                        )
  {
   const size_t i = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t variables = get_global_size(1);
//---
   if(i == 0)
     {
      const int step = variables * 3;
      const int start = v * 3 + 2;
      float last = 0;
      d_forw[v] = 0;
      for(int p = 1; p < total; p++)
        {
         float m = concat_inp[start + p * step];
         d_forw[p * variables + v] = last = 1 + (1 - m) * last;
        }
     }
   else
     {
      const int step = -(variables * 3);
      const int start = (total - 1) * variables + v * 3 + 2;
      float last = 0;
      d_bakw[(total - 1) + v] = 0;
      for(int p = 1; p < total; p++)
        {
         float m = concat_inp[start + p * step];
         d_bakw[(total - 1 - p) * variables + v] = last = 1 + (1 - m) * last;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float2 Rotate(const float x, const float cos_theta, const float sin_theta)
  {
   float2 result = 0;
   result.s0 = cos_theta + x * sin_theta;
   result.s1 = x * cos_theta - sin_theta;
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void HiVTPrepare(__global const float *data,
                          __global float2 *output
                         )
  {
   const size_t t = get_global_id(0);
   const size_t v = get_global_id(1);
   const size_t total_v = get_global_size(1);
//---
   const int shift_data = t * total_v;
   const int shift_out = shift_data * total_v;
//---
   float value = data[shift_data + v + total_v] - data[shift_data + v];
   const float theta = atan(value);
   const float cos_theta = cos(theta);
   const float sin_theta = sin(theta);
   const float2 main = Rotate(value, cos_theta, sin_theta);
//---
   for(int a = 0; a < total_v; a++)
     {
      float2 o = main;
      if(a != v)
         o -= Rotate(data[shift_data + a + total_v] - data[shift_data + a], cos_theta, sin_theta);
      output[shift_out + a] = o;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void GateElementMul(__global const float *inputs1,
                             __global const float *inputs2,
                             __global const float *gate,
                             __global float *out
                            )
  {
   const int i = get_global_id(0);
//---
   const float g = gate[i];
   float result = g * inputs1[i] + (1 - g) * inputs2[i];
   if(isnan(result) || isinf(result))
      result = 0;
//---
   out[i] = result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void GateElementMulGrad(__global const float *inputs1,
                                 __global float *inputs1_gr,
                                 __global const float *inputs2,
                                 __global float *inputs2_gr,
                                 __global const float *gate,
                                 __global float *gate_gr,
                                 __global const float *out_gr,
                                 const int activ1,
                                 const int activ2,
                                 const int activ_gate
                                )
  {
   const int i = get_global_id(0);
//---
   const float g = gate[i];
   const float i1 = inputs1[i];
   const float i2 = inputs1[i];
   const float grad = out_gr[i];
//---
   float i1_gr = grad * g;
   if(isnan(i1_gr) || isinf(i1_gr))
      i1_gr = 0;
//---
   float i2_gr = grad * (1 - g);
   if(isnan(i2_gr) || isinf(i2_gr))
      i2_gr = 0;
//---
   float g_gr = grad * (i1 - i2);
   if(isnan(g_gr) || isinf(g_gr))
      i1_gr = 0;
//---
   inputs1_gr[i] = Deactivation(i1_gr, i1, activ1);
   inputs2_gr[i] = Deactivation(i2_gr, i2, activ2);
   gate_gr[i] = Deactivation(g_gr, g, activ_gate);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void TransposeRCD(__global const float *matrix_in, ///<[in] Input matrix
                           __global float *matrix_out ///<[out] Output matrix
                          )
  {
   const int r = get_global_id(0);
   const int c = get_global_id(1);
   const int d = get_global_id(2);
   const int rows = get_global_size(0);
   const int cols = get_global_size(1);
   const int dimension = get_global_size(2);
//---
   matrix_out[(c * rows + r)*dimension + d] = matrix_in[(r * cols + c) * dimension + d];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void OrthoganalLoss(__global const float *data,
                             __global float *grad,
                             const int add
                            )
  {
   const size_t r = get_global_id(0);
   const size_t c = get_local_id(1);
   const size_t cols = get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   uint ls = min((uint)cols, (uint)LOCAL_ARRAY_SIZE);
//---
   const int shift1 = r * cols + c;
   const int shift2 = c * cols + r;
   float value1 = data[shift1];
   float value2 = (shift1 == shift2 ? value1 : data[shift2]);
   if(isinf(value1) || isnan(value1))
      value1 = 0;
   if(isinf(value2) || isnan(value2))
      value2 = 0;
   float v2 = value1 * value2;
   if(isinf(v2) || isnan(v2))
      v2 = 0;
   for(int i = 0; i < cols; i += ls)
     {
      //---
      if(i <= c && (i + ls) > c)
         Temp[c - i] = (i == 0 ? 0 : Temp[c - i]) + v2;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   uint count = min(ls, (uint)cols);
   do
     {
      count = (count + 1) / 2;
      if(c < ls)
         Temp[c] += (c < count && (c + count) < cols ? Temp[c + count] : 0);
      if(c + count < ls)
         Temp[c + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   const float sum = Temp[0];
   float loss = -pow((float)(r == c) - sum, 2.0f);
   float g = (2 * (sum - (float)(r == c))) * loss;
   g = 2 * value2 * g;
   if(isinf(g) || isnan(g))
      g = 0;
   if(add == 1)
      grad[shift1] += g;
   else
      grad[shift1] = g;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CalcDistance(__global const float *data,
                           __global float *distance,
                           const int dimension
                          )
  {
   const size_t main = get_global_id(0);
   const size_t slave = get_local_id(1);
   const int total = (int)get_local_size(1);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
   int ls = min((int)total, (int)LOCAL_ARRAY_SIZE);
//---
   const int shift_main = main * dimension;
   const int shift_slave = slave * dimension;
   const int shift_dist = main * total + slave;
//--- calc distance
   float dist = 0;
   if(main != slave)
     {
      for(int d = 0; d < dimension; d++)
         dist += pow(data[shift_main + d] - data[shift_slave + d], 2.0f);
     }
//--- Look Max
   for(int i = 0; i < total; i += ls)
     {
      if(!isinf(dist) && !isnan(dist))
        {
         if(i <= slave && (i + ls) > slave)
            Temp[slave - i] = max((i == 0 ? 0 : Temp[slave - i]), dist);
        }
      else
         if(i == 0)
            Temp[slave] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
//---
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(slave < count && (slave + count) < ls)
        {
         if(Temp[slave] < Temp[slave + count])
            Temp[slave] = Temp[slave + count];
         Temp[slave + count] = 0;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- Normalize
   if(Temp[0] > 0)
      dist /= Temp[0];
   if(isinf(dist) || isnan(dist))
      dist = 1;
//--- result
   distance[shift_dist] = dist;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void FeedForwardLocalMax(__global const float *matrix_i,
                                  __global const float *distance,
                                  __global float *matrix_o,
                                  const float radius
                                 )
  {
   const size_t i = get_global_id(0);
   const size_t total = get_global_size(0);
   const size_t d = get_global_id(1);
   const size_t dimension = get_global_size(1);
//---
   const int shift_dist = i * total;
   const int shift_out = i * dimension + d;
//---
   float result = -3.402823466e+38;
   for(int k = 0; k < total; k++)
     {
      if(distance[shift_dist + k] > radius)
         continue;
      int shift = k * dimension + d;
      result = max(result, matrix_i[shift]);
     }
   matrix_o[shift_out] = result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CalcInputGradientLocalMax(__global const float *matrix_i,
                                        __global float *matrix_ig,
                                        __global const float *distance,
                                        __global const float *matrix_o,
                                        __global const float *matrix_g,
                                        const float radius
                                       )
  {
   const size_t i = get_global_id(0);
   const size_t total = get_global_size(0);
   const size_t d = get_global_id(1);
   const size_t dimension = get_global_size(1);
//---
   float result = 0;
   float value = matrix_i[i * dimension + d];
   for(int k = 0; k < total; k++)
     {
      if(distance[k * total + i] > radius)
         continue;
      int shift = k * dimension + d;
      if(fabs(matrix_o[shift] - value) <= 1.192092896e-07f)
         result += matrix_g[shift];
     }
   matrix_ig[i * dimension + d] = result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MHMaskAttentionOut(__global const float *q,      ///<[in] Matrix of Querys
                                 __global const float *kv,     ///<[in] Matrix of Keys
                                 __global float *score,        ///<[out] Matrix of Scores
                                 __global const float *mask,   ///<[in] Mask Matrix
                                 __global float *out,          ///<[out] Matrix of attention
                                 const int dimension,          ///< Dimension of Key
                                 const int heads_kv,
                                 const float mask_level
                                )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_k = dimension * (2 *  heads_kv * k + h_kv);
   const int shift_v = dimension * (2 *  heads_kv * k + heads_kv + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k;
   const bool b_mask = (mask[shift_s] < mask_level);
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- sum of exp
   uint count = 0;
   if(k < ls)
     {
      temp[k] = 0;
      do
        {
         if(b_mask || q_id >= (count * ls + k))
            if((count * ls) < (kunits - k))
              {
               float sum = 0;
               int sh_k = 2 * dimension * heads_kv * count * ls;
               for(int d = 0; d < dimension; d++)
                  sum = q[shift_q + d] * kv[shift_k + d + sh_k];
               sum = exp(sum / koef);
               if(isnan(sum))
                  sum = 0;
               temp[k] = temp[k] + sum;
              }
         count++;
        }
      while((count * ls + k) < kunits);
     }
   barrier(CLK_LOCAL_MEM_FENCE);
   count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k < ls)
         temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
      if(k + count < ls)
         temp[k + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- score
   float sum = temp[0];
   float sc = 0;
   if(b_mask || q_id >= (count * ls + k))
      if(sum != 0)
        {
         for(int d = 0; d < dimension; d++)
            sc = q[shift_q + d] * kv[shift_k + d];
         sc = exp(sc / koef) / sum;
         if(isnan(sc))
            sc = 0;
        }
   score[shift_s] = sc;
   barrier(CLK_LOCAL_MEM_FENCE);
//--- out
   for(int d = 0; d < dimension; d++)
     {
      uint count = 0;
      if(k < ls)
         do
           {
            if((count * ls) < (kunits - k))
              {
               int sh_v = 2 * dimension * heads_kv * count * ls;
               float sum =
                  kv[shift_v + d + sh_v] * (count == 0 ? sc : score[shift_s + count * ls]);
               if(isnan(sum))
                  sum = 0;
               temp[k] = (count > 0 ? temp[k] : 0) + sum;
              }
            count++;
           }
         while((count * ls + k) < kunits);
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k < ls)
            temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
         if(k + count < ls)
            temp[k + count] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      out[shift_q + d] = temp[0];
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MHMaskAttentionInsideGradients(__global const float *q, __global float *q_g,
      __global const float *kv, __global float *kv_g,
      __global const float *mask, __global float *mask_g,
      __global const float *scores, __global const float *gradient,
      const int kunits, const int heads_kv, const float mask_level)
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (q_id * heads + h) * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
      for(int v = q_id; v < kunits; v += qunits)
        {
         float grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v;
            for(int g = 0; g < qunits; g++)
               grad += gradient[shift_g + dimension * (hq - h + g  * heads)] *
                       scores[shift_score + g * step_score];
           }
         int shift_v = dimension * (2 *  heads_kv * v + heads_kv + h) + d;
         kv_g[shift_v] = grad;
        }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = (heads_kv + h_kv) * dimension + d;
   int shift_key = h_kv * dimension + d;
   for(int k = 0; k < kunits; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k];
      if(sc == 0)
         continue;
      for(int v = 0; v < kunits; v++)
         sc_g += scores[shift_s + v] * out_g * kv[shift_val + 2 * v * heads_kv * dimension] *
                 ((float)(k == v) - sc);
      grad += sc_g * kv[shift_key + 2 * k * heads_kv * dimension];
     }
   q_g[shift_q] = grad / koef;
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      for(int k = q_id; k < kunits; k += qunits)
        {
         int shift_k = dimension * (2 *  heads_kv * k + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq++)
           {
            int shift_score = hq * kunits + k;
            float val = kv[shift_k + heads_kv * dimension];
            for(int scr = 0; scr < qunits; scr++)
              {
               float sc_g = 0;
               int shift_sc = scr * kunits * heads;
               float sc = scores[shift_sc + k];
               if(sc == 0)
                  continue;
               for(int v = 0; v < kunits; v++)
                  sc_g += scores[shift_sc + v] * gradient[shift_g + scr * dimension] *
                          val * ((float)(k == v) - sc);
               grad += sc_g * q[shift_q + scr * dimension];
              }
           }
         kv_g[shift_k] = grad / koef;
        }
     }
//--- Mask's gradient
   for(int k = q_id; k < kunits; k += qunits)
     {
      float m = mask[shift_s + k];
      if(m < mask_level)
         mask_g[shift_s + k] = 0;
      else
         mask_g[shift_s + k] = 1 - m;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void CalcPositionBias(__global const float *data1,
                               __global const float *data2,
                               __global float *result,
                               const int dimension
                              )
  {
   const size_t idx1 = get_global_id(0);
   const size_t idx2 = get_global_id(1);
   const size_t total1 = get_global_size(0);
   const size_t total2 = get_global_size(1);
//---
   const int shift1 = idx1 * dimension;
   const int shift2 = idx2 * dimension;
   const int shift_out = idx1 * total2 + idx2;
//---
   float res = 0;
   for(int i = 0; i < dimension; i++)
      res = pow(data1[shift1 + i] - data2[shift2 + i], 2.0f);
   res = sqrt(res);
   res = exp(-res);
   if(isnan(res) || isinf(res))
      res = 0;
//---
   result[shift_out] = res;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MHPosBiasAttentionOut(__global const float *q,         ///<[in] Matrix of Querys
                                    __global const float *k,         ///<[in] Matrix of Keys
                                    __global const float *v,         ///<[in] Matrix of Values
                                    __global float *score,           ///<[out] Matrix of Scores
                                    __global const float *pos_bias,  ///<[in] Position Bias
                                    __global float *out,             ///<[out] Matrix of attention
                                    const int dimension,             ///< Dimension of Key
                                    const int heads_kv,
                                    const int use_pos_bias
                                   )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k_id = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_kv = dimension * (heads_kv * k_id + h_kv);
   const int shift_s = kunits * (q_id *  heads + h) + k_id;
   const int shift_pb = q_id * kunits + k_id;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- sum of exp
   uint count = 0;
   if(k_id < ls)
     {
      temp[k_id] = 0;
      do
        {
         if(q_id >= (count * ls + k_id))
            if((count * ls) < (kunits - k_id))
              {
               float sum = 0;
               int sh_k = dimension * heads_kv * count * ls;
               for(int d = 0; d < dimension; d++)
                  sum = q[shift_q + d] * k[shift_kv + d + sh_k];
               sum = exp(sum / koef);
               if(isnan(sum))
                  sum = 0;
               temp[k_id] = temp[k_id] + sum + (use_pos_bias > 0 ? pos_bias[shift_pb + count * ls] : 0);
              }
         count++;
        }
      while((count * ls + k_id) < kunits);
     }
   barrier(CLK_LOCAL_MEM_FENCE);
   count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k_id < ls)
         temp[k_id] += (k_id < count && (k_id + count) < kunits ? temp[k_id + count] : 0);
      if(k_id + count < ls)
         temp[k_id + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- score
   float sum = temp[0];
   float sc = 0;
   if(q_id >= (count * ls + k_id))
      if(sum != 0)
        {
         for(int d = 0; d < dimension; d++)
            sc = q[shift_q + d] * k[shift_kv + d];
         sc = (exp(sc / koef) + (use_pos_bias > 0 ? pos_bias[shift_pb] : 0)) / sum;
         if(isnan(sc))
            sc = 0;
        }
   score[shift_s] = sc;
   barrier(CLK_LOCAL_MEM_FENCE);
//--- out
   for(int d = 0; d < dimension; d++)
     {
      uint count = 0;
      if(k_id < ls)
         do
           {
            if((count * ls) < (kunits - k_id))
              {
               int sh_v = 2 * dimension * heads_kv * count * ls;
               float sum =
                  v[shift_kv + d + sh_v] * (count == 0 ? sc : score[shift_s + count * ls]);
               if(isnan(sum))
                  sum = 0;
               temp[k_id] = (count > 0 ? temp[k_id] : 0) + sum;
              }
            count++;
           }
         while((count * ls + k_id) < kunits);
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k_id < ls)
            temp[k_id] += (k_id < count && (k_id + count) < kunits ? temp[k_id + count] : 0);
         if(k_id + count < ls)
            temp[k_id + count] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      out[shift_q + d] = temp[0];
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MHPosBiasAttentionInsideGradients(__global const float *q, __global float *q_g,
      __global const float *k, __global float *k_g,
      __global const float *v, __global float *v_g,
      __global const float *scores, __global const float *gradient,
      const int kunits, const int heads_kv)
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int h_kv = h % heads_kv;
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (q_id * heads + h) * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   if(h < heads_kv)
      for(int v_id = q_id; v_id < kunits; v_id += qunits)
        {
         float grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + v_id;
            for(int g = 0; g < qunits; g++)
               grad += gradient[shift_g + dimension * (hq - h + g  * heads)] *
                       scores[shift_score + g * step_score];
           }
         int shift_v = dimension * (heads_kv * v_id + h) + d;
         v_g[shift_v] = grad;
        }
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = h_kv * dimension + d;
   int shift_key = h_kv * dimension + d;
   for(int k_id = 0; k_id < kunits; k_id++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k_id];
      if(sc == 0)
         continue;
      for(int v_id = 0; v_id < kunits; v_id++)
         sc_g += scores[shift_s + v_id] * out_g * v[shift_val + v_id * heads_kv * dimension] *
                 ((float)(k_id == v_id) - sc);
      grad += sc_g * k[shift_key + k_id * heads_kv * dimension];
     }
   q_g[shift_q] = grad / koef;
//--- Calculating Key's gradients
   if(h < heads_kv)
     {
      for(int k_id = q_id; k_id < kunits; k_id += qunits)
        {
         int shift_k = dimension * (heads_kv * k_id + h_kv) + d;
         grad = 0;
         for(int hq = h; hq < heads; hq += heads_kv)
           {
            int shift_score = hq * kunits + k_id;
            float val = v[shift_k];
            for(int scr = 0; scr < qunits; scr++)
              {
               float sc_g = 0;
               int shift_sc = scr * kunits * heads;
               float sc = scores[shift_sc + k_id];
               if(sc == 0)
                  continue;
               for(int v_id = 0; v_id < kunits; v_id++)
                  sc_g += scores[shift_sc + v_id] * gradient[shift_g + scr * dimension] *
                          val * ((float)(k_id == v_id) - sc);
               grad += sc_g * q[shift_g + scr * heads * dimension];
              }
           }
         k_g[shift_k] = grad / koef;
        }
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void DiversityLoss(__global const float *data,
                            __global float *grad,
                            const int activation,
                            const int add
                           )
  {
   const size_t main = get_global_id(0);
   const size_t slave = get_local_id(1);
   const size_t dim = get_local_id(2);
   const size_t total = get_local_size(1);
   const size_t dimension = get_local_size(2);
//---
   __local float Temp[LOCAL_ARRAY_SIZE];
//---
   const int shift_main = main * dimension + dim;
   const int shift_slave = slave * dimension + dim;
//---
   const int value_main = data[shift_main];
   const int value_slave = data[shift_slave];
   float delt = value_main - value_slave;
//---
   for(int d = 0; d < dimension; d++)
     {
      for(int i = 0; i < total; i += LOCAL_ARRAY_SIZE)
        {
         if(d == dim)
           {
            if(i <= slave && (i + LOCAL_ARRAY_SIZE) > slave)
              {
               int k = i % LOCAL_ARRAY_SIZE;
               float val = pow(delt, 2.0f) / total;
               if(isinf(val) || isnan(val))
                  val = 0;
               Temp[k] = ((d == 0 && i == 0) ? 0 : Temp[k]) + val;
              }
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
     }
//---
   const int ls = min((int)total, (int)LOCAL_ARRAY_SIZE);
   int count = ls;
   do
     {
      count = (count + 1) / 2;
      if(slave < count)
        {
         Temp[slave] += ((slave + count) < ls ? Temp[slave + count] : 0);
         if(slave + count < ls)
            Temp[slave + count] = 0;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//---
   float loss = exp(-Temp[0]);
   float gr = 2 * pow(loss, 2.0f) * delt / total;
   if(isnan(gr) || isinf(gr))
      gr = 0;
//---
   for(int d = 0; d < dimension; d++)
     {
      for(int i = 0; i < total; i += LOCAL_ARRAY_SIZE)
        {
         if(d == dim)
           {
            if(i <= slave && (i + LOCAL_ARRAY_SIZE) > slave)
              {
               int k = i % LOCAL_ARRAY_SIZE;
               Temp[k] = ((d == 0 && i == 0) ? 0 : Temp[k]) + gr;
              }
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      //---
      int count = ls;
      do
        {
         count = (count + 1) / 2;
         if(slave < count && d == dim)
           {
            Temp[slave] += ((slave + count) < ls ? Temp[slave + count] : 0);
            if(slave + count < ls)
               Temp[slave + count] = 0;
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      if(slave == 0 && d == dim)
        {
         if(isnan(Temp[0]) || isinf(Temp[0]))
            Temp[0] = 0;
         if(add > 0)
            grad[shift_main] += Deactivation(Temp[0], value_main, activation);
         else
            grad[shift_main] = Deactivation(Temp[0], value_main, activation);
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MHRelativeAttentionOut(__global const float *q,         ///<[in] Matrix of Querys
                                     __global const float *k,         ///<[in] Matrix of Keys
                                     __global const float *v,         ///<[in] Matrix of Values
                                     __global const float *bk,        ///<[in] Matrix of Positional Bias Keys
                                     __global const float *bv,        ///<[in] Matrix of Positional Bias Values
                                     __global const float *gc,        ///<[in] Global content bias vector
                                     __global const float *gp,        ///<[in] Global positional bias vector
                                     __global float *score,           ///<[out] Matrix of Scores
                                     __global float *out,             ///<[out] Matrix of attention
                                     const int dimension              ///< Dimension of Key
                                    )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k_id = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (q_id * heads + h);
   const int shift_kv = dimension * (heads * k_id + h);
   const int shift_gc = dimension * h;
   const int shift_s = kunits * (q_id *  heads + h) + k_id;
   const int shift_pb = q_id * kunits + k_id;
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
//--- score
   float sc = 0;
   for(int d = 0; d < dimension; d++)
     {
      float val_q = q[shift_q + d];
      float val_k = k[shift_kv + d];
      float val_bk = bk[shift_kv + d];
      sc += val_q * val_k + val_q * val_bk + val_k * val_bk + gc[shift_q + d] * val_k + gp[shift_q + d] * val_bk;
     }
   sc = exp(sc / koef);
   if(isnan(sc) || isinf(sc))
      sc = 0;
//--- sum of exp
   for(int cur_k = 0; cur_k < kunits; cur_k += ls)
     {
      if(k_id >= cur_k && k_id < (cur_k + ls))
        {
         int shift_local = k_id % ls;
         temp[shift_local] = (cur_k == 0 ? 0 : temp[shift_local]) + sc;
        }
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   uint count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k_id < ls)
         temp[k_id] += (k_id < count && (k_id + count) < kunits ? temp[k_id + count] : 0);
      if(k_id + count < ls)
         temp[k_id + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
//--- score
   float sum = temp[0];
   if(isnan(sum) || isinf(sum) || sum <= 1e-6f)
      sum = 1;
   sc /= sum;
   score[shift_s] = sc;
   barrier(CLK_LOCAL_MEM_FENCE);
//--- out
   for(int d = 0; d < dimension; d++)
     {
      float val_v = v[shift_kv + d];
      float val_bv = bv[shift_kv + d];
      float val = sc * (val_v + val_bv);
      if(isnan(val) || isinf(val))
         val = 0;
      //--- sum of value
      for(int cur_v = 0; cur_v < kunits; cur_v += ls)
        {
         if(k_id >= cur_v && k_id < (cur_v + ls))
           {
            int shift_local = k_id % ls;
            temp[shift_local] = (cur_v == 0 ? 0 : temp[shift_local]) + val;
           }
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k_id < count && (k_id + count) < kunits)
            temp[k_id] += temp[k_id + count];
         if(k_id + count < ls)
            temp[k_id + count] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      if(k_id == 0)
         out[shift_q + d] = (isnan(temp[0]) || isinf(temp[0]) ? 0 : temp[0]);
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
__kernel void MHRelativeAttentionInsideGradients(__global const float *q, __global float *q_g,
      __global const float *k, __global float *k_g,
      __global const float *v, __global float *v_g,
      __global const float *bk, __global float *bk_g,
      __global const float *bv, __global float *bv_g,
      __global const float *gc, __global float *gc_g,
      __global const float *gp, __global float *gp_g,
      __global const float *scores,
      __global const float *gradient,
      const int kunits
                                                )
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
   const int shift_q = dimension * (q_id * heads + h) + d;
   const int shift_s = (q_id * heads + h) * kunits;
   const int shift_g = h * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
//--- Calculating Value's gradients
   int step_score = kunits * heads;
   for(int v_id = q_id; v_id < kunits; v_id += qunits)
     {
      float grad = 0;
      int shift_score = h * kunits + v_id;
      for(int g = 0; g < qunits; g++)
         grad += gradient[shift_g + dimension * (g  * heads)] *
                 scores[shift_score + g * step_score];
      int shift_v = dimension * (heads * v_id + h) + d;
      if(isnan(grad) || isinf(grad))
         grad = 0;
      v_g[shift_v] = grad;
      bv_g[shift_v] = grad;
     }
//--- Calculating Query's gradients
   float grad_gc = 0;
   float grad_gp = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = h * dimension + d;
   int shift_key = h * dimension + d;
   for(int k_id = 0; k_id < kunits; k_id++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k_id];
      if(sc == 0)
         continue;
      for(int v_id = 0; v_id < kunits; v_id++)
         sc_g += scores[shift_s + v_id] * out_g *
                 (v[shift_val + v_id * heads * dimension] + bv[shift_val + v_id * heads * dimension]) *
                 ((float)(k_id == v_id) - sc);
      grad_gc += sc_g * k[shift_key + k_id * heads * dimension];
      grad_gp += sc_g * bk[shift_key + k_id * heads * dimension];
     }
//---
   if(isinf(grad_gc) || isnan(grad_gc))
      grad_gc = 0;
   if(isinf(grad_gp) || isnan(grad_gp))
      grad_gp = 0;
//---
   q_g[shift_q] = (grad_gc + grad_gp) / koef;
   gc_g[shift_q] = grad_gc / koef;
   gp_g[shift_q] = grad_gp / koef;
//--- Calculating Key's gradients
   for(int k_id = q_id; k_id < kunits; k_id += qunits)
     {
      int shift_k = dimension * (heads * k_id + h) + d;
      float grad = 0;
      float grad_bk = 0;
      int shift_score = h * kunits + k_id;
      float val = (v[shift_k] + bv[shift_k]);
      for(int scr = 0; scr < qunits; scr++)
        {
         float sc_g = 0;
         int shift_sc = scr * kunits * heads;
         float sc = scores[shift_sc + k_id];
         if(sc == 0)
            continue;
         for(int v_id = 0; v_id < kunits; v_id++)
            sc_g += scores[shift_sc + v_id] * gradient[shift_g + scr * dimension] *
                    val * ((float)(k_id == v_id) - sc);
         float _q = q[shift_g + scr * heads * dimension];
         grad += sc_g * (_q + bk[shift_k] + gc[shift_g + scr * heads * dimension]);
         grad_bk += sc_g * (_q + k[shift_k] + gp[shift_g + scr * heads * dimension]);
        }
      k_g[shift_k] = grad / koef;
      bk_g[shift_k] = grad_bk / koef;
     }
  }
//+------------------------------------------------------------------+
