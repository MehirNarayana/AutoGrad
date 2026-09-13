# AutoGrad

AutoGrad is an educational automatic differentiation and tensor engine written from scratch in
C++17. It supports multidimensional tensors, reverse-mode autodiff, broadcasting, tiled batched
matrix multiplication, basic neural network layers, SGD, and inference-only model serialization.

> [!NOTE]
> AutoGrad is an experimental learning project. The current kernels are CPU-only and single-threaded.

## Inspiration

This project began after watching Andrej Karpathy's
["The spelled-out intro to neural networks and backpropagation: building micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0)
and studying [micrograd](https://github.com/karpathy/micrograd).

The first version recreated the core idea in C++: every scalar value stored its data, gradient,
parents, and local backward function. Operations constructed a computation graph, and a reverse
topological traversal propagated derivatives through it.

Once that scalar engine worked, the project started moving beyond a direct micrograd style
implementation. Scalar nodes were replaced by multidimensional tensors, and the focus shifted to
the problems a tensor engine must solve: flattening nested input, shape and stride metadata,
broadcasting, matrix multiplication, transposed views, tensor-valued gradients, layers, optimizers,
and model persistence.


## Current features

- Multidimensional `Tensor<T>` construction from flat or nested `std::vector` data
- Shape and stride metadata
- Stride-based transpose operations
- NumPy-style broadcasting through effective zero strides
- Broadcasted tensor addition
- Scalar addition and multiplication
- Tiled, batched matrix multiplication
- Reverse-mode automatic differentiation
- Gradient accumulation through broadcasted operations
- Backward implementations for matmul, addition, scalar operations, transpose, `tanh`, softmax,
  and negative log likelihood
- `Linear`, `Tanh`, `Softmax`, and `Sequential` layers
- Mean cross-entropy loss through softmax plus NLL
- SGD with `step()` and `zeroGrad()`
- Inference-only binary model saving and loading
- End-to-end MNIST MLP training and image inference example

Supported tensor scalar types are:

- `float`
- `double`
- `std::int32_t`
- `std::int64_t`

Only supported floating-point tensors can track gradients. Integer tensors are intended for data
such as class indices and should be constructed with gradient tracking disabled.

## Architecture

```text
Tensor<T>                         public, lightweight handle
    |
    +-- shared_ptr<TensorImpl<T>> storage, shape, strides, gradients
                                      |
                                      +-- parent tensors
                                      +-- local backward callback

Layer<T>
    |
    +-- Linear<T>
    +-- Tanh<T>
    +-- Softmax<T>
    +-- Sequential<T> -------- owns child layers with unique_ptr
```


## Requirements

- A C++17 compiler
- CMake 4.1 or newer, matching the current project configuration
- Internet access the first time the optional MNIST example downloads its dataset

The core library has no third-party runtime or numerical-library dependencies.

## Building

Build the library only:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Build the library and MNIST example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUTOGRAD_BUILD_EXAMPLES=ON

cmake --build build -j
```


The MNIST executable is created at:

```text
build/Examples/MNIST/examples
```

## Basic usage

```cpp
#include <Layers/Activations.hpp>
#include <Layers/Linear.hpp>
#include <Layers/LossFns/CrossEntropy.hpp>
#include <Layers/Sequential.hpp>
#include <Optimizer.hpp>
#include <Tensor.hpp>

#include <cstdint>
#include <utility>
#include <vector>

Sequential<float> model{
    Linear<float>{2, 8},
    Activations::Tanh<float>{},
    Linear<float>{8, 2},
};

Optimizers::SGD<float> optimizer{model.parameters(), 0.01F};
Loss::CrossEntropy lossFunction;

Tensor<float> inputs{
    std::vector<float>{0.0F, 1.0F, 1.0F, 0.0F},
    {2, 2},
    false
};

Tensor<std::int64_t> targets{
    std::vector<std::int64_t>{1, 0},
    {2},
    false
};

optimizer.zeroGrad();
Tensor<float> logits = model.forward(std::move(inputs));
Tensor<float> loss = lossFunction.forward(logits, targets);
loss.backward();
optimizer.step();
```

Parameterless `backward()` requires the output tensor to contain one element. A non-scalar output
would require an explicit upstream gradient, which is not currently exposed by the public API.

## MNIST example

The example trains this MLP:

```text
784 inputs -> Linear(784, 128) -> Tanh -> Linear(128, 10)
```

The built-in training configuration uses five epochs, a batch size of 64, and a learning rate of
0.05. The dataset is downloaded and extracted into `Examples/MNIST/dataset` automatically when the
example target is built.

Train and save the model:

```bash
./build/Examples/MNIST/examples train
```

The default model path is:

```text
Examples/MNIST/mnist.model
```

An alternative output path can be supplied:

```bash
./build/Examples/MNIST/examples train my-model.bin
```


### Extracting an MNIST image

### Running inference

```bash
./build/Examples/MNIST/examples infer \
  Examples/MNIST/mnist.model \
  {path_to_pgm_image}
```

Inference currently accepts 28x28 grayscale PGM images (`P5` or `P2`).

## Model format

`ModelWriter` recursively saves the model's layer types, scalar types, tensor shapes, weights, and
biases. `ModelReader::loadModel<T>()` reconstructs the layer tree with gradient tracking disabled.
The files are intended for inference, not for resuming training: gradients, optimizer state, and the
autograd graph are not stored.


## Formatting

The repository contains a `.clang-format` configuration. Format all C++ headers and sources with:

```bash
find include src Examples -type f \( -name '*.hpp' -o -name '*.cpp' \) \
  -exec clang-format -i {} +
```

## Limitations and future directions

- Add SIMD and multithreaded CPU kernels
- Add CUDA, Metal programmed operations
- Add more tensor operations, activations, losses, optimizers, and layers
- Move toward runtime dtype/device dispatch if a non-templated public tensor API becomes desirable
- Add pytorch like dataloaders so data preperation becomes easier
