#include <Tensor.hpp>
#include <stdexcept>
#include <utility>

TensorBase::TensorBase(){};
TensorBase::TensorBase(std::vector<size_t> dataShape): dim(dataShape.size()), dataShape(std::move(dataShape)), stride(dim){};

void TensorBase::fillStride(size_t lastIndex){
    int accum = 1;
    for (int i = lastIndex; i>=0; i--){
        stride[i] = accum;
        accum = dataShape[i] * accum;
    }
}


size_t TensorBase::getDim(){
    return dim;
}

std::vector<size_t> TensorBase::getShape(){
    return dataShape;
}

void TensorBase::swapStride(size_t dim1, size_t dim2){
    try{
        std::swap(stride[dim1], stride[dim2]);
    }
    catch (...){
        throw std::runtime_error{"error"};
    }
}

void TensorBase::swapShape(size_t dim1, size_t dim2){
    try{
        std::swap(dataShape[dim1], dataShape[dim2]);
    }
    catch (...){
        throw std::runtime_error{"error"};
    }
}

