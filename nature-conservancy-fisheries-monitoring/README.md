# Installation Guide for Keras with Theano backend

* Python version: Keras is compatible with Python 2.7 - 3.4. So you can choose the version based on your personal preferences. I would suggest to download Anaconda because it has its own package manager and many packages already installed for you. Anaconda is available for windows, linux and mac. (https://www.continuum.io/Downloads). You can install it in your home directory if you are working on a remote machine where you have no administrator privileges like Emma on her mpi cluster. The Python version of Anaconda also was originally compiled in such a way that you should have nearly no struggle with Keras and Theano. I would recommend to install the Anaconda version which ships with Python 2.7. 

* Most of the tutorials regarding Keras are written for version < 2.0.1. The last version before the release of version 2 is 1.2.2, so I would suggest to stick to Keras v. 1.2.2. If you are following along the keras tutorial about classifying cats and dogs then you have to stick to version 1.2.2 anyways.

* Keras sits on top of Tensorflow or Theano. You can choose for yourself which backend better fits your needs. Tensorflow is a product of Google. It has some minor benefits like being capable of using the Inception model which you cannot load with Theano or being able to use multiple GPUs at once. I experienced lesser difficulties with Theano and no big drawbacks in using it as a backend. 

* Theano and Tensorflow are sitting on top of CUDA and are using cuDNN for performing Deep Neural Network calculations. So it is a requirement that you also install these packages before using Tensorflow or Theano. 

* CUDA: https://developer.nvidia.com/cuda-downloads

* cuDNN: https://developer.nvidia.com/cudnn

* If you are on Windows and have errors regarding loading some dll files when using Theano, then I would suggest the following approach to install Theano again (even on linux or mac, this seems a good approach to follow):
	* ``` conda install mingw libpython ``` (on mac or linux, you can probably skip mingw)
	* ``` conda install theano pygpu ``` (by installing pygpu, you also get the new libgpuarray package which seems to replace the cuda backend in the future)

* After that you can install Keras by using pip (as long as the pip version is linked to your anaconda python installation)
	* ``` pip install keras==1.2.2 ```

* You have to tell Theano to use the GPU instead of the CPU. Therefore, you have to create a .theanorc file in your home directory. It should at least contain the following flags:
	``` [global] 
	floatX = float32 
	device = gpu

	[lib]
	cnmem = 0.6 
	```