# convolutional_neural_network
~~~cs
// 1. cnn declaration
int startDimension = 28; // or (int) sqrt(784)
int[] _cnn = { 1, 32, 64 }; // non-RGB = 1 or RGB = 3 for start dimension
int[] _filter = {  5,  5 }; // x and y dim
int[] _stride = {  2,  2 }; // replaces pooling with higher strides than 1 
~~~

Then...

~~~cs
  for (int l = 0; l < cnn_layerLen; l++) // cnn layers : 2
  {
      int inLen = _cnn[l], outLen = _cnn[l + 1], cd = _dim[l], fd = _filter[l], st = _stride[l]; 
      // source non rgb + cmaps : 1,32,64 // conv dim x and y : src 28 -> 24,12 // filter dim x and y : 5,5 // stride : 2,2
      int csIn = _cs[l], csOut = _cs[l + 1], ks = _ks[l], bs = _bs[l], inDmSq = _inDim[l] * _inDim[l]; // steps = convolutions in - out, kernels, bias
      int csOut2 = _csIn[l + 1], csIn2 = _csIn[l + 0];
      for (int im = 0; im < inLen; im++) // input maps
      {
          int cIn = csIn2 + im * inDmSq; // set start and current input maps
          int b = bs; int f = ks;
          for (int om = 0, c = csOut2; om < outLen; om++, b++, f+= fd * fd) // output maps
          {
              float bias = _b[b];
              for (int i = 0; i < cd; i += st) // conv dim y col
                  for (int j = 0; j < cd; j += st, c++, cIn++) // conv dim x row
                  {
                      float sum = bias; // add bias first
                      for (int col = 0; col < fd; col++) // filter dim y cols
                          for (int rs = 0, cfs = col * fd; rs < fd; rs++) // filter dim x rows                                              
                              sum += _c[cIn + cfs + rs] * _f[f + cfs + rs];  // float ff = _f[f + cfs + rs]; float cc = _c[cIn + cfs + rs];
                      _c[c] = sum > 0 ? sum : 0; // relu activation for each feature 
                  }
          }
      }
  } // end cnn layer
~~~

## Intro

## Before Convolutions



## Effekt of filters

## A simple CNN

## A better CNN

## Padding

## Max Pooling

## Naiv Backpropagation

## Flipped Kernel Filter

## FF + BP + Update

## A Serios Approach

## Summary

