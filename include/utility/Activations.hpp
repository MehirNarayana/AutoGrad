#include "../Tensor.hpp"
template <typename scalarType>

class Tanh{
    template <typename scalarType>
    Tensor<scalarType> forward(Tensor<scalarType>& inputTensor){
        inputTensor.tanh();
    }
};