# Neural Network from Scratch

This repository is part of the coursework for the Introduction to Artificial Intelligence course of [Novo Ensino Suplementar](https://novoensinossuplementar.com). The primary goal of this project is to build a neural network from scratch, implementing fundamental components such as neurons, activation functions, forward propagation, backpropagation, and gradient descent without relying on high-level libraries like TensorFlow or PyTorch. This project serves as a comprehensive guide to understanding the theoretical and mathematical foundations of neural networks, as well as their practical implementation.

## Table of Contents

- [Introduction](#introduction)
- [What is a Neural Network?](#what-is-a-neural-network)
- [Neurons and Layers](#neurons-and-layers)
  - [Neurons](#neurons)
  - [Layers](#layers)
- [Weights and Biases](#weights-and-biases)
  - [Weights](#weights)
  - [Biases](#biases)
- [Activation Functions](#activation-functions)
  - [Sigmoid](#sigmoid)
  - [ReLU](#relu)
  - [Softmax](#softmax)
- [Forward Propagation](#forward-propagation)
- [Cost Function](#cost-function)
- [Backpropagation](#backpropagation)
- [Gradient Descent](#gradient-descent)
- [Training the Neural Network](#training-the-neural-network)
- [Epochs](#epochs)
- [Example with Digit Recognition](#example-with-digit-recognition)
- [Conclusion](#conclusion)

---

### Introduction

Welcome to the "Neural Network from Scratch" project! The primary goal of this project is to build a neural network from the ground up, without using high-level machine learning libraries like TensorFlow or PyTorch. By doing so, we aim to provide a deep understanding of the inner workings of neural networks, including the theoretical and mathematical rigor behind them.

### What is a Neural Network?

A neural network is a computational model inspired by the human brain, composed of layers of interconnected nodes or neurons. These networks are capable of learning complex patterns and functions by adjusting the connections (weights) between neurons based on the data they are trained on. It's not a linear model because we have [activation functions](#activation-functions) to introduce non-linearity. Because of this, a neural network can learn more complex patterns than a linear model.

### Neurons and Layers

#### Neurons

Neurons are the fundamental building blocks of a neural network, functioning as computational units that process input data and produce output signals. Each neuron receives one or more input values, applies a weight to each input, sums the results, adds a bias, and then passes this sum through an activation function to produce the output.

Mathematically, a neuron's operation can be expressed as:

$$ z = \displaystyle \sum_{i=1}^{n} w_i \times x_i + b $$

where $ z $ is the weighted sum of the inputs plus the bias, each $ w_i $ and $ x_i $ are the weights and input values respectively, and $ b $ is the bias.

After the activation function is applied to the weighted sum, the output of the neuron is the result of the operation, as follows:

$$ a = \sigma(z) $$

where $ \sigma $ is the activation function and the result $ a $ is the output of the neuron.

> We'll discuss _bias_ and _activation functions_ in the [weights and biases](#weights-and-biases) and [activation functions](#activation-functions) sections respectively.

#### Layers

Layers refer to collections of neurons arranged sequentially. Each layer transforms the output of the previous layer through a series of weighted sums and activation functions.

> We'll explain the mathematical workings in the [forward propagation](#forward-propagation) section.

There are three main types of layers in a neural network:

- **Input layer:** The first layer that directly receives the raw input data. This layer does not perform any operations; it simply passes the data to the next layer. For many types of problems, the input data is represented as vectors. Each data point is a vector of features. In our example ([Digit Recognition](#example-with-digit-recognition)), the input data is a set of images where each image is a $ 28 \times 28 $ pixel grid, which is represented as a vector of length $ 784 = 28 \times 28 $ entries of pixel intensity.

- **Hidden layer:** Intermediate layers located between the input and output layers. These layers perform the bulk of the computation and feature extraction. Each hidden layer consists of neurons that apply weights, biases, and activation functions to the inputs received from the previous layer. Hidden layers are crucial for capturing complex patterns in the input data.

- **Output layer:** The final layer that produces the network's predictions or outputs. The number of neurons in this layer typically corresponds to the number of classes in a classification problem or the dimensionality of the target variable in a regression problem. The activation function used in this layer depends on the nature of the problem (e.g., [softmax](#softmax) for multi-class classification, and [sigmoid](#sigmoid) for binary classification).

### Weights and Biases

#### Weights

Are numerical values associated with the **connections** between neurons. They determine the strenght of these connections and, in turn, the influence that one neurons's input. Think of weights as the coefficients that adjust the **impact** of incoming data. They can increase or decrease the _importance_ of specific features in the input data.

Weights are fundementally tied to the connections between neurons rather tha the neurons themselves because they **represent how the output of one neuron affects the input of another**. Each wight adjusts the influece of a specific input feature or the output from a previous layer on the next neuron.

In mathematical terms, if we have two neurons $ N_i $ and $ N_j $, with $ N_i $ providing an output $ a_i $ and $ N_j $ receiving this output as an input, the weight $ w_{ij} $ determines how strongly $ a_i $ affects the input to $ N_j $. The weighted sum of inputs into $ N_j $ can be expressed as:

$$ z_j = \displaystyle \sum_{i} w_{ij} \times a_i + b_j $$

where $ z*j $ is the input to neuron $ N_j $ **before** applying the activation function, $ w*{ij} $ is the weight associated with the **connection** from $ N_i $ to $ N_j $, $ a_i $ is the activation (output) of $ N_i $, and $ b_j $ is the bias of the $ N_j $ neuron.

#### Biases

_Coming soon_

### Activation Functions

_Coming soon_

#### Sigmoid

The sigmoid function maps any real-valued number into the range (0, 1), making it useful for binary classification.

$ \sigma(z) = \frac{1}{1 + e^{-z}} $

#### ReLU

The Rectified Linear Unit (ReLU) function is widely used in hidden layers due to its simplicity and efficiency.

$ \text{ReLU}(z) = \max(0, z) $

#### Softmax

The softmax function converts a vector of values into a probability distribution, often used in the output layer for multi-class classification.

$ \text{softmax}(z*i) = \frac{e^{z_i}}{\sum*{j} e^{z_j}} $

### Forward Propagation

_Coming soon_

### Cost Function

_Coming soon_

### Backpropagation

_Coming soon_

### Gradient Descent

_Coming soon_

### Training the Neural Network

_Coming soon_

### Epochs

_Coming soon_

### Example with Digit Recognition

_Coming soon_

### Conclusion

_Coming soon_
