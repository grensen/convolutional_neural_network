
# Basic Convolutional Neuronal Network With 99,2 Accuracy On MNIST in 5 Minutes Using C#

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

### The Demo

<p align="center">
  <img src="https://github.com/grensen/convolutional_neural_network/blob/main/figures/cnn_demo.png?raw=true">
</p>

To run the demo program, you must have VisualStudio2022 installed on your machine. Then just start a console application with DotNet 6, copy the code and change from debug to release mode and run the demo. MNIST data and network are then managed by the code. This line `AutoData d = new(@"C:_mnist\");` specifies where the MNIST dataset is stored. To do this, the data is simply loaded from my github on first use. On future starts the data will be loaded from the directory where it was saved.

Lets take a look at the demo...
The CNN in its current form has 3 hyperparameters. 

The convolution describes the maps used for each layer. The first layer creates 8 new output maps from the input map through 8 kernel filters. For this a 5 x 5 kernel is used. Additionally a stride of 2 is used to reduce the map size. Each of the eight maps has a dimension of 12 and a resolution of 144 pixels.

These 8 feature maps build on the next layer the input maps which now form with 8 * 16 kernels with a filter size of 3 x 3 to create 16 output maps. Each of the 8 * 144 input maps are now "fully connected" with one kernel to one output map. Thus, 16 new output maps are created with a dimension of 10 and a resolution of 100. The output layer thus has 1600 neurons, which are then sent to the neural network.

You may wonder why I built the network this way. Actually, the reason is essentially due to performance reasons.

The larger 5 x 5 kernel on layer 1 worked very well, but is relatively expensive to compute. The stride of 2 cuts this calculation massively and also reduces the output resolution. Only 8 kernels are used here. The 8 input maps that generate 16 output maps already use 128 filters. Since the size of the filter reduces the calculation time, I have reduced the size here, the stride remains 1.

After the network is ready to start, the training for 20 epochs begins. This means that the entire MNIST dataset of 60,000 examples is simply trained 20 times. All this happens on my system in 5 minutes. However, it could take a little longer depending on the computer or laptop used, but it should be possible to estimate this due to the epoch-by-epoch output. 

Of course, other network designs are possible, but the one presented worked best among those I tested. Alternatively, I could have used a stride 2 on layer 2 to create more output maps. However, my demo might have ended up at 99.1% or taken 10 minutes.


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

