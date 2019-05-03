<center>

![](https://i.imgur.com/Fr293ZF.jpg)

> Deep Learning in raw performance, or so I hope.
</center>

### Warning: Heavy Construction

## About
TensorBolt is a work in progress computation graph and deep neural network library supporting multiple backends, where only CPU backend is working at the moment. The library is built with future and external improvements in minds allowing maximum flexibility to using GPU, multiple GPUs, run on cluster, etc. It's all up to you. Just fork it and do what you want to do.

## Actual About

This is a computational graph library written in C. Unlike cgraph, there is no Lua API at the moment as my major focus is to build a stable C API and improve the design of the original library, mainly separating `ndarray` package from `tb_graph` API which is now abstracted over any NDArray implementation (could be OpenCL, CUDA, ...)

## Setup and build

My development environment is OSX Mojave, but the library should build fine with minor CMakeLists changes.

Simply

```
$ git clone https://github.com/tensorbolt/TensorBolt
$ cd source
$ mkdir build
$ cd build
$ cmake ..
```

I do not care for now about Windows Support.

## API
You check the API here: [https://tensorbolt.praisethemoon.org](https://tensorbolt.praisethemoon.org), or Wiki for some tutorials.

 