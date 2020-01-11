# DLNetwork
Transparent deep learning package

`DLNetworkLayer.m` contains the description of the base class implementing a single deep network layer

`stepForward.m` and `stepBackward.m` perform forward- and backpropagation for a single `DLNetworkLayer` layer

`train.m` and `test.m` implent training and testing of an example network (LeNet on the MNIST dataset)

`loadMNISTImages.m` and `loadMNISTLabels.m` are the MNIST dataset import functions downloaded elsewhere
