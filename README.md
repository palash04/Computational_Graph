# Deep Neural Networks Using Computational Graph
This project was made as the final project for E0 251 - Data Structrues and Algorithms course in Spring 2021 at Indian Institute of Science (IISc) Bangalore, India.

#### -- Project Timeline: [April 15, 2021 - May 30, 2021]
#### -- Project Status: [Completed]

## Abstract
The basic idea in a computational graph is to express some neural network model as a directed graph expressing a sequence of computational steps. 
Each step in the sequence corresponds to a vertex in the computational graph, 
where each vertex takes some inputs and produces some output as a function of its inputs. 
In this project, I have built such a computational graph for a feedforward neural network model, which achieves two tasks: Forward Pass, and Backward Pass. 
To create such a computational graph I have made use of graph data structure and for implementing forward and backward passes, 
I have used a topological ordering algorithm. 
Forward pass is the first phase where we get predictions as output, and backward pass is the second phase where we use the outputs to update weights in the model moving backward in the graph so as to get better predictions in the next iteration, 
and the process of forward and backward pass keeps continuing until we achieve our target value.

## Prerequisites
* GCC Compiler 10.0+ version

### Tech Stack
* C++
* Graph Data Structure
* Kahn's Algorithm for Topological Ordering
* Forward Propagation
* Backward Propagation


## Dataset Description
A dummy dataset was used with 9 dimensional feature set and 10 number of classes.

## Results:
![Input](https://user-images.githubusercontent.com/26361028/128869265-0efa1e46-605c-41be-a182-8ecba2d7fe5d.png)
![Target](https://user-images.githubusercontent.com/26361028/128869274-a7fbf902-e843-4dbe-b8c4-e661e7d47a52.png)

In the above image we can see that the model is updating its parameters over the epochs in order to make the accurate prediction.

## Authors:
* Palash Mahendra Kamble - [palash04](https://github.com/palash04/)
