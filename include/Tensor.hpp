#pragma once

#include <ScalarType.hpp>
#include <TensorImpl.hpp>
#include <vector>
template <typename scalarType = float>
class Tensor {
    static_assert(isSupportedTensorScalarType<scalarType>, "Tensor scalar type is not supported");

private:
    template <typename>
    friend class Tensor;

    std::shared_ptr<TensorImpl<scalarType>> impl;

    explicit Tensor(std::shared_ptr<TensorImpl<scalarType>> inputImpl)
        : impl(std::move(inputImpl)) {}

public:
    template <typename inputType>
    Tensor(const inputType& data, bool shouldTrackGradient = true)
        : impl(std::make_shared<TensorImpl<scalarType>>(data, shouldTrackGradient)) {}

    Tensor(std::vector<scalarType> inputVector,
           std::vector<size_t> inputShape,
           bool shouldTrackGradient = true)
        : impl(std::make_shared<TensorImpl<scalarType>>(
              std::move(inputVector), std::move(inputShape), shouldTrackGradient)) {}

    size_t getDim() const noexcept {
        return impl->getDim();
    }

    const std::vector<size_t>& getShape() const noexcept {
        return impl->getShape();
    }

    const std::vector<size_t>& getStride() const noexcept {
        return impl->getStride();
    }

    size_t getNumTotalElements() const noexcept {
        return impl->getNumTotalElements();
    }

    scalarType operator[](size_t index) const {
        return (*impl)[index];
    }

    Tensor<scalarType> transpose(size_t dim1, size_t dim2) {
        return Tensor<scalarType>(impl->transpose(dim1, dim2));
    }

    template <typename otherScalarType>
    Tensor<std::common_type_t<scalarType, otherScalarType>>
    operator+(const Tensor<otherScalarType>& other) {
        using resultType = std::common_type_t<scalarType, otherScalarType>;
        return Tensor<resultType>((*impl) + other.impl);
    }

    template <typename otherScalarType>
    Tensor<std::common_type_t<scalarType, otherScalarType>>
    operator*(const Tensor<otherScalarType>& other) {
        using resultType = std::common_type_t<scalarType, otherScalarType>;
        return Tensor<resultType>((*impl) * other.impl);
    }

    template <typename otherScalarType,
              std::enable_if_t<std::is_arithmetic_v<otherScalarType>, int> = 0>
    Tensor<std::common_type_t<scalarType, otherScalarType>> operator+(otherScalarType other) {
        using resultType = std::common_type_t<scalarType, otherScalarType>;
        return Tensor<resultType>((*impl) + other);
    }

    template <typename otherScalarType,
              std::enable_if_t<std::is_arithmetic_v<otherScalarType>, int> = 0>
    Tensor<std::common_type_t<scalarType, otherScalarType>> operator*(otherScalarType other) {
        using resultType = std::common_type_t<scalarType, otherScalarType>;
        return Tensor<resultType>((*impl) * other);
    }

    Tensor<scalarType> tanh() {
        return Tensor<scalarType>(impl->tanh());
    }

    Tensor<scalarType> softmax() {
        return Tensor<scalarType>(impl->softmax());
    }

    void step(scalarType learningRate) {
        impl->step(learningRate);
    }

    void zeroGrad() {
        impl->zeroGrad();
    }

    template <typename anyType>
    Tensor<scalarType> NLLLoss(Tensor<anyType>& target) {
        return Tensor<scalarType>(impl->NLLLoss(target.impl));
    }

    void backward() {
        impl->backwardPass();
    }

    std::vector<scalarType> getData() {
        return impl->getDataCopy();
    }
};
