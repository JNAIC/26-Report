
# Weekly learning summary
- Partly Implemented AutoGrad system in Pytorch to help understand its mechanism
- Implemented a simple CNN(similiar to LeNet) with Pytorch
- Redesigned AlexNet to fit CIFAR10, a subset of ImageNet, and implemented it with pytorch
	- its acc:![[alexnet_result.png]]
	- Though it takes me lots of time and energy, I think that it is meaningful as it is my first model actually designed by me to a certain extent. I absorbed the design philosophy of AlexNet, which I think is big channels, fast downsampling in the first Conv and deep convolution layers(compared with networks before 2012)