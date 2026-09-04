#include <Tensor.hpp>
#include <stdexcept>
#include <utility>

TensorBaseImpl::TensorBaseImpl() {};
TensorBaseImpl::TensorBaseImpl(std::vector<size_t> dataShape)
    : dim(dataShape.size()), dataShape(std::move(dataShape)), stride(dim) {};

void TensorBaseImpl::fillStride(size_t lastIndex) {
    size_t accum = 1;
    for (size_t reverseIndex = lastIndex + 1; reverseIndex > 0; --reverseIndex) {
        const size_t i = reverseIndex - 1;
        stride[i] = accum;
        accum = dataShape[i] * accum;
    }
}

size_t TensorBaseImpl::getDim() const noexcept {
    return dim;
}

const std::vector<size_t>& TensorBaseImpl::getShape() const noexcept {
    return dataShape;
}

const std::vector<size_t>& TensorBaseImpl::getStride() const noexcept {
    return stride;
}

void TensorBaseImpl::swapStride(size_t dim1, size_t dim2) {
    try {
        std::swap(stride[dim1], stride[dim2]);
    } catch (...) {
        throw std::runtime_error{"error"};
    }
}

void TensorBaseImpl::swapShape(size_t dim1, size_t dim2) {
    try {
        std::swap(dataShape[dim1], dataShape[dim2]);
    } catch (...) {
        throw std::runtime_error{"error"};
    }
}

size_t TensorBaseImpl::getNumTotalElements() const noexcept {
    return numTotalElements;
}

void TensorBaseImpl::topoSort(TensorBaseImpl* root,
                              std::unordered_set<TensorBaseImpl*>& visited,
                              std::vector<TensorBaseImpl*>& topoList) {
    if (visited.find(root) == visited.end()) {
        visited.insert(root);
        for (const auto& parent : root->parents) {
            topoSort(parent.get(), visited, topoList);
        }
        topoList.push_back(root);
    }
}

void TensorBaseImpl::applyBackward() {
    std::unordered_set<TensorBaseImpl*> visited;
    std::vector<TensorBaseImpl*> topoList;
    TensorBaseImpl::topoSort(this, visited, topoList);
    for (size_t reverseIndex = topoList.size(); reverseIndex > 0; --reverseIndex) {
        const size_t i = reverseIndex - 1;
        if (topoList[i]->backward) {
            topoList[i]->backward();
        }
    }
}
