### Basic Convolutional neural network

<p align="center">
  <img src="https://github.com/grensen/convolutional_neural_network/blob/main/figures/convolution_meaning.png?raw=true">
</p>

Quite intimidating when you look for the meaning of convolution. The idea behind this article is to make this idea of convolutional neural networks simple as possible.
A convolutional neural network is simply a neural network with a convolutional network on top of it. The application area of CNNs is mainly in image recognition for 2-dimensional images. The basic idea is just to print new images with stamps, which are often called filters or even better kernels.

### The Convolution Step

<p align="center">
  <img src="https://github.com/grensen/convolutional_neural_network/blob/main/figures/convolution_explainer.gif?raw=true">
</p>

The animation shows all the magic of convolution. Initially, the filter consists of randomly selected weights. During training, these weights are formed according to the direction of the network target, extracting features that allow these networks to achieve much higher accuracy. 

### The Pooling Step

<p align="center">
  <img src="https://github.com/grensen/convolutional_neural_network/blob/main/figures/max_pooling_explainer.gif?raw=true">
</p>

The convolution is often followed by the pooling step. Actually, pooling is not necessary, but since a convolutional network creates many of these feature maps, the computational effort is so enormous that it is common practice to reduce the resolution of the outputs. In addition to the max pooling shown in the animation, there is also average pooling and other types of pooling.

Unlike the common practice, however, I do not use a pooling technique here. Instead, a stride of 2 is used in the convolution, which produces the same output map.

### Make It Simple

<p align="center">
  <img src="https://github.com/grensen/convolutional_neural_network/blob/main/figures/NN_vs._CNN_ji.png?raw=true">
</p>

In my mind, neural networks and convolutional networks are quite similar. The NN has an input neuron on the left with its weight connected to an output neuron on the right. The CNN follows the same connection pattern. Only that it is a whole map of neurons that are connected with a kernel filter consisting of several weights to generate the output map. 



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

