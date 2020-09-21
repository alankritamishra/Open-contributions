# RNN IN PYTORCH

##  What is RNN
A recurrent neural network (RNN) is a type of artificial neural network commonly used in speech recognition and natural language processing (NLP). 
**RNNs are designed to recognize a data's sequential characteristics and use patterns to predict the next likely scenario.**
Recurrent Neural Networks(RNNs) have been the answer to most problems dealing with sequential data and Natural Language Processing(NLP) problems for many years.

## Why RNN ?
Humans don’t start their thinking from scratch every second. As you are reading this article , you understand each word based on your understanding of previous words.
You don’t throw everything away and start thinking from scratch again. Your thoughts have persistence, or we learn from experience

Traditional neural networks can’t do this, and it seems like a major shortcoming.
For ex. You want to judge the tone of a sentence you'll sense it with the help of your prior knowledge, but for a tradational neural network it's unclear to predict the outcome 
of a sentence without having any prior knowledge about it.

Recurrent neural networks address this issue. They are networks with loops in them, allowing information to persist.
![1_lQ4izz9ZbhKYD8NClZpsmQ](https://user-images.githubusercontent.com/66169287/93784314-7fcad480-fc4a-11ea-8ad6-2a59b771ff1b.png)

**Recurrent Neural Networks have loops.**

## Feed forward v/s RNN
The main difference is in how the input data is taken in by the model.
![Slide3-1 1](https://user-images.githubusercontent.com/66169287/93785166-67a78500-fc4b-11ea-9d8e-b9a7b1e64b1b.jpg)
Traditional feed-forward neural networks take in a fixed amount of input data all at the same time and produce a fixed amount of output each time. On the other hand,
**RNNs do not consume all the input data at once. Instead, they take them in one at a time and in a sequence.** At each step, the RNN does a series of calculations before producing an output. 
The output, known as the hidden state, is then combined with the next input in the sequence to produce another output. 
This process continues until the model is programmed to finish or the input sequence ends.
As we can see, the calculations at each time step consider the context of the previous time steps in the form of the hidden state. Being able to use this contextual information from previous inputs is the key essence to RNNs’ success in sequential problems.

While it may seem that a different RNN cell is being used at each time step in the graphics, the underlying principle of **Recurrent Neural Networks is that the RNN cell is actually the exact same one and reused throughout.**
## Working of RNN
![RNN-unrolled 1](https://user-images.githubusercontent.com/66169287/93785682-ff0cd800-fc4b-11ea-9165-7b278e753908.png)
This is an unrolled version of RNN.

A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Each network produces an output called a hidden state, which is feed forwarded to the next network which allows RNNs to deal with sequences

### How RNN learn??
A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. Each network produces an output called a hidden state, which is feed forwarded to the next network which allows RNNs to deal with sequences

Each hidden state is calculated as-
***hidden(t)= F(hidden(t-1),input(t))***

Each hidden output is calculated with the previous hidden output and the current input.

In the first step, a hidden state will usually be seeded as a matrix of zeros, so that it can be fed into the RNN cell together with the first input in the sequence. In the simplest RNNs, the hidden state and the input data will be multiplied with weight matrices

The result of these multiplications will then be passed through an activation function(such as a tanh function) to introduce non-linearity
***hidden(t)= tanh(weight(hidden)* hidden(t-1) + weight(input)*input(t))***

The hidden state that we just produced will then be fed back into the RNN cell together with the next input and this process continues until we run out of input or the model is programmed to stop producing output

![rnn-2 1](https://user-images.githubusercontent.com/66169287/93787802-44caa000-fc4e-11ea-8d07-bda7450e377a.gif)

## Training in RNN
During training, for each piece of training data we’ll have a “correct answer” that we want the model to output. During inital rounds when we input our data, we won’t obtain outputs that are equal to these correct answers. However, after receiving these outputs, what we’ll do during training is that we’ll calculate the loss of that process, which measures how far off the model’s output is from the correct answer. Using this loss, we can calculate the gradient of the loss function for back-propagation.
With the gradient that we just obtained, we can update the weights in the model accordingly so that future computations with the input data will produce more accurate results. The weight here refers to the weight matrices that are multiplied with the input data and hidden states during the forward pass. This entire process of calculating the gradients and updating the weights is called back-propagation. Combined with the forward pass, back-propagation is looped over and again, allowing the model to become more accurate with its outputs each time as the weight matrices values are modified to pick out the patterns of the data.
ll of the weights are actually the same as that RNN cell is essentially being re-used throughout the process.
![rnn-bptt-with-gradients 1](https://user-images.githubusercontent.com/66169287/93791838-5c0b8c80-fc52-11ea-98f9-9ce6fe96c977.png)

## Dealing with Textual data
 To feed textual data into RNN we have to convert our data into a set of numbers, such as embeddings, one-hot encodings, etc. such that the network can parse the data better
 Also we need to pad each sequence of input so that RNNs recieve same length of input sequence.
 
 ## Drawbacks of RNN
 1. **The problem of Long Term Depedency**
 
 One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task.
 Cases where we need more context. Consider trying to predict the last word in the text “the movie is releasing today it's a nice weather, we shall go to the movies” Recent information suggests that the next word is probably movies, but if we want to narrow down it , we need the context of movie, from further back. 
 **It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.**


Unfortunately, as that gap grows, ***RNNs become unable to learn to connect the information.***

## What is Pytorch
![Pytorch 1](https://user-images.githubusercontent.com/66169287/93798095-bad50400-fc5a-11ea-9b2d-11eab62604e9.png)
PyTorch is a Python-based scientific computing package that uses the power of graphics processing units. 
It is also one of the preferred deep learning research platforms built to provide maximum flexibility and speed.
It is known for providing two of the most high-level features; namely, tensor computations with strong GPU acceleration support and building deep neural networks on a tape-based autograd systems.
It can be used to build neural networks effortlessly
PyTorch is a native Python package by design. Its functionalities are built as Python classes, hence all its code can seamlessly integrate with Python packages and modules. Similar to NumPy, this Python-based library enables GPU-accelerated tensor computations plus provides rich options of APIs for neural network applications. PyTorch provides a complete end-to-end research framework which comes with the most common building blocks for carrying out everyday deep learning research. It allows chaining of high-level neural network modules because it supports Keras-like API in its torch.nn package.

## Why Pytorch for NLP??
There are multiple benefits of using Pytorch for deep learning some of them are
1. Easy debugging
2. Data Parallelism:  PyTorch can distribute computational work among multiple CPU or GPU cores.
3. Dynamic Computational Graph Support: the network behavior can be changed programmatically at runtime. This facilitates more efficient model optimization 
4. Open Neural Network Exchange support: developers can easily move models between different tools and choose the combination that work best for them and their given use case.
---
### Reasons to use Pytorch for NLP specific tasks
1. ***Dealing with Out of Vocabulary words***: During infrence we might come across words that are not in trained vocabullary of model, these are k/a Out of Vocabulary words, PyTorch supports a cool feature that replaces the rare words in our training data with Unknown token. This, in turn, helps us in tackling the problem of Out of Vocabulary words.
2. ***Handling Variable Length sequences*** : As mentioned above RNNs are capable of taking fixed length sequences only,
*Padding is a process of adding an extra token called padding token at the beginning or end of the sentence. As the number of the words in each sentence varies, we convert the variable length input sentences into sentences with the same length by adding padding tokens.*
**Packed padding ignores the input timesteps with padding token. These values are never shown to the Recurrent Neural Network which helps us in building a dynamic Recurrent Neural Network**
3.***Wrappers and Pretrained models***: The state of the art architectures are being launched for PyTorch framework. Hugging Face released Transformers which provides more than 32 state of the art architectures for the Natural Language Understanding Generation!

## IMPLEMENTAION OF RNN
In the notebook we'll see how to implement a RNN model, in Pytorch and use it for Spam classification



