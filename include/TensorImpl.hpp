#pragma once

#include <ScalarType.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

template <typename scalarType>
class TensorImpl;

class TensorBaseImpl {
private:
    static void topoSort(TensorBaseImpl* root,
                         std::unordered_set<TensorBaseImpl*>& visited,
                         std::vector<TensorBaseImpl*>& topoList);

protected:
    size_t dim = 0;
    std::vector<size_t> dataShape;
    std::vector<size_t> stride;
    size_t numTotalElements = 0;

    void fillStride(size_t lastIndex);

    void swapStride(size_t dim1, size_t dim2);
    void swapShape(size_t dim1, size_t dim2);

    std::function<void()> backward;

    std::vector<std::shared_ptr<TensorBaseImpl>> parents;
    std::unordered_set<std::shared_ptr<TensorBaseImpl>> visited;
    std::vector<std::shared_ptr<TensorBaseImpl>> topoList;
    void applyBackward();

    TensorBaseImpl();
    TensorBaseImpl(std::vector<size_t> dataShape);

public:
    virtual ~TensorBaseImpl() = default;
    size_t getDim() const noexcept;
    const std::vector<size_t>& getShape() const noexcept;
    const std::vector<size_t>& getStride() const noexcept;
    size_t getNumTotalElements() const noexcept;
};

template <typename scalarType = float>
class TensorImpl : public TensorBaseImpl,
                   public std::enable_shared_from_this<TensorImpl<scalarType>> {
    static_assert(isSupportedTensorScalarType<scalarType>,
                  "TensorImpl scalar type is not supported");

private:
    template <typename>
    friend class TensorImpl;

    std::vector<scalarType> data;

    static constexpr size_t tileM = 32;
    static constexpr size_t tileN = 32;
    static constexpr size_t tileK = 32;

    std::vector<scalarType>& getData() {
        return data;
    }

    template <typename inputType>
    struct isVector : std::false_type {};

    template <typename inputType, typename Alloc>
    struct isVector<std::vector<inputType, Alloc>> : std::true_type {};

    bool isScalarType(scalarType x) {
        return true;
    }

    template <typename randomType>
    bool isScalarType(randomType x) {
        return false;
    }

    template <typename inputType>
    size_t findShapeAndFlatten(const inputType& data, size_t currDepth) {
        if constexpr (!isVector<inputType>::value) {
            if (isScalarType(data)) {
                this->data.push_back(data);
                return 1;
            } else {
                throw std::runtime_error{
                    "Data inside tensor does not match type supplied to TensorImpl"};
            }
        } else {
            std::optional<size_t> nextDimSize;
            size_t currDimSize = data.size();
            if (dim <= currDepth) {
                dim++;
                dataShape.push_back(currDimSize);
            }

            for (size_t i = 0; i < currDimSize; i++) {
                size_t currSize = findShapeAndFlatten(data[i], currDepth + 1);
                if (nextDimSize) {
                    if (nextDimSize != currSize) {
                        throw std::runtime_error{"Not a rectangular tensor"};
                    }
                } else {
                    nextDimSize = currSize;
                }
            }
            return currDimSize;
        }
    }

    template <typename otherScalarType>
    void broadcast(TensorImpl<otherScalarType>& other,
                   std::vector<size_t>& newShape,
                   std::vector<size_t>& effectiveStrideCurr,
                   std::vector<size_t>& effectiveStrideOther,
                   size_t& batchDim,
                   size_t& totalElements,
                   size_t ignoredDimensions) {
        const size_t otherDim = other.getDim();
        const size_t currDim = dim;

        if (otherDim < ignoredDimensions || currDim < ignoredDimensions) {
            throw std::runtime_error{"Not enough dimensions to broadcast"};
        }

        batchDim = std::max(otherDim - ignoredDimensions, currDim - ignoredDimensions);
        newShape.resize(batchDim);
        effectiveStrideCurr.resize(batchDim);
        effectiveStrideOther.resize(batchDim);
        totalElements = 1;

        std::vector<size_t> otherShape = other.getShape();
        const std::vector<size_t>& otherStride = other.getStride();
        const size_t currBatchDim = currDim - ignoredDimensions;
        const size_t otherBatchDim = otherDim - ignoredDimensions;

        const size_t commonBatchDim = std::min(currBatchDim, otherBatchDim);

        for (size_t offset = 0; offset < commonBatchDim; ++offset) {
            const size_t outputIndex = batchDim - 1 - offset;
            const size_t currIndex = currBatchDim - 1 - offset;
            const size_t otherIndex = otherBatchDim - 1 - offset;
            const size_t currSize = dataShape[currIndex];
            const size_t otherSize = otherShape[otherIndex];

            if (currSize == otherSize) {
                newShape[outputIndex] = currSize;
                effectiveStrideCurr[outputIndex] = stride[currIndex];
                effectiveStrideOther[outputIndex] = otherStride[otherIndex];
            } else if (currSize == 1) {
                newShape[outputIndex] = otherSize;
                effectiveStrideCurr[outputIndex] = 0;
                effectiveStrideOther[outputIndex] = otherStride[otherIndex];
            } else if (otherSize == 1) {
                newShape[outputIndex] = currSize;
                effectiveStrideCurr[outputIndex] = stride[currIndex];
                effectiveStrideOther[outputIndex] = 0;
            } else {
                throw std::runtime_error("Tensors cannot be broadcast together");
            }

            totalElements *= newShape[outputIndex];
        }

        for (size_t offset = commonBatchDim; offset < currBatchDim; ++offset) {
            const size_t outputIndex = batchDim - 1 - offset;
            const size_t currIndex = currBatchDim - 1 - offset;

            newShape[outputIndex] = dataShape[currIndex];
            effectiveStrideCurr[outputIndex] = stride[currIndex];
            effectiveStrideOther[outputIndex] = 0;
            totalElements *= newShape[outputIndex];
        }

        for (size_t offset = commonBatchDim; offset < otherBatchDim; ++offset) {
            const size_t outputIndex = batchDim - 1 - offset;
            const size_t otherIndex = otherBatchDim - 1 - offset;

            newShape[outputIndex] = otherShape[otherIndex];
            effectiveStrideCurr[outputIndex] = 0;
            effectiveStrideOther[outputIndex] = otherStride[otherIndex];
            totalElements *= newShape[outputIndex];
        }
    }

    template <typename otherScalarType>
    std::shared_ptr<TensorImpl<std::common_type_t<scalarType, otherScalarType>>>
    tiledMatmul(const std::shared_ptr<TensorImpl<otherScalarType>>& other,
                std::vector<size_t> newShape,
                const std::vector<size_t>& effectiveStrideCurr,
                const std::vector<size_t>& effectiveStrideOther,
                size_t totalElements,
                size_t mCurr,
                size_t nCurr,
                size_t nOther,
                size_t batchDim) {
        const size_t elementsPerBatch = mCurr * nOther;
        const size_t totalBatches = totalElements / elementsPerBatch;

        using resultType = std::common_type_t<scalarType, otherScalarType>;

        std::vector<resultType> result(totalElements);
        size_t currRowStride = stride[dim - 2];
        size_t currColStride = stride[dim - 1];

        const size_t otherDim = other->getDim();
        const std::vector<size_t>& otherStride = other->getStride();
        size_t otherRowStride = otherStride[otherDim - 2];
        size_t otherColStride = otherStride[otherDim - 1];

        for (size_t batchIndex = 0; batchIndex < totalBatches; ++batchIndex) {
            size_t batchIndexCopy = batchIndex;
            size_t currCoordinate = 0;
            size_t otherCoordinate = 0;
            for (size_t reverseIndex = batchDim; reverseIndex > 0; --reverseIndex) {
                const size_t index = reverseIndex - 1;
                const size_t currDimStrideCoefficient = batchIndexCopy % newShape[index];
                batchIndexCopy /= newShape[index];
                currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
            }

            const size_t batchStartIndex = batchIndex * elementsPerBatch;
            const size_t mNew = newShape[batchDim];
            const size_t nNew = newShape[batchDim + 1];
            const size_t kNew = nCurr;
            for (size_t currVertical = 0; currVertical < mNew; currVertical += TensorImpl::tileM) {
                for (size_t otherHorizontal = 0; otherHorizontal < nNew;
                     otherHorizontal += TensorImpl::tileN) {
                    for (size_t k = 0; k < kNew; k += tileK) {
                        const size_t currHorizontal = k;
                        const size_t otherVertical = k;
                        const size_t endCurrHorizontal = std::min(k + tileK, kNew);
                        const size_t endCurrVertical =
                            std::min(currVertical + TensorImpl::tileM, mNew);
                        const size_t endOtherHorizontal =
                            std::min(otherHorizontal + TensorImpl::tileN, nNew);

                        for (size_t y = currVertical; y < endCurrVertical; ++y) {
                            for (size_t xOther = otherHorizontal; xOther < endOtherHorizontal;
                                 ++xOther) {
                                resultType accum = 0;
                                for (size_t index = 0; index < endCurrHorizontal - currHorizontal;
                                     ++index) {
                                    const size_t actualCurrCoordinate =
                                        currCoordinate + y * currRowStride +
                                        (currHorizontal + index) * currColStride;
                                    const size_t actualOtherCoordinate =
                                        otherCoordinate + (otherVertical + index) * otherRowStride +
                                        xOther * otherColStride;
                                    resultType multipliedResult =
                                        static_cast<resultType>(data[actualCurrCoordinate]) *
                                        static_cast<resultType>((*other)[actualOtherCoordinate]);
                                    accum += multipliedResult;
                                }
                                const size_t targetCoordinate = batchStartIndex + y * nNew + xOther;
                                result[targetCoordinate] += accum;
                            }
                        }
                    }
                }
            }
        }

        bool outputTracksGradient = trackGradient || other->trackGradient;
        std::shared_ptr<TensorImpl<resultType>> output = std::make_shared<TensorImpl<resultType>>(
            std::move(result), std::move(newShape), outputTracksGradient);
        if (outputTracksGradient) {
            output->parents = {this->shared_from_this(), other};
        }
        std::weak_ptr<TensorImpl<scalarType>> currWeak{this->shared_from_this()};
        std::weak_ptr<TensorImpl<otherScalarType>> otherWeak{other};
        std::weak_ptr<TensorImpl<resultType>> outputWeak{output};

        output->backward = [batchDim,
                            currWeak,
                            otherWeak,
                            outputWeak,
                            effectiveStrideCurr = std::move(effectiveStrideCurr),
                            effectiveStrideOther = std::move(effectiveStrideOther),
                            totalBatches,
                            elementsPerBatch]() {
            std::shared_ptr<TensorImpl<resultType>> output = outputWeak.lock();
            std::shared_ptr<TensorImpl<scalarType>> curr = currWeak.lock();
            std::shared_ptr<TensorImpl<otherScalarType>> other = otherWeak.lock();

            if (!output || !curr || !other) {
                return;
            }

            std::vector<resultType>& outputGradientVector = output->gradient->getData();
            const std::vector<size_t>& outputShape = output->getShape();
            const std::vector<size_t>& outputStride = output->getStride();
            const size_t outputDim = output->getDim();
            size_t outputRowStride = outputStride[outputDim - 2];
            size_t outputColStride = outputStride[outputDim - 1];

            size_t currRowStride = curr->stride[curr->dim - 2];
            size_t currColStride = curr->stride[curr->dim - 1];

            const std::vector<size_t>& otherShape = other->getShape();
            const std::vector<size_t>& otherStride = other->getStride();
            const size_t otherDim = other->getDim();
            size_t otherRowStride = otherStride[otherDim - 2];
            size_t otherColStride = otherStride[otherDim - 1];

            for (size_t batchIndex = 0; batchIndex < totalBatches; ++batchIndex) {
                size_t batchIndexCopy = batchIndex;
                size_t currCoordinate = 0;
                size_t otherCoordinate = 0;

                for (size_t reverseIndex = batchDim; reverseIndex > 0; --reverseIndex) {
                    const size_t index = reverseIndex - 1;
                    const size_t currDimStrideCoefficient = batchIndexCopy % outputShape[index];
                    batchIndexCopy /= outputShape[index];
                    currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                    otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
                }

                const size_t batchStartIndex = batchIndex * elementsPerBatch;

                if (curr->trackGradient) {
                    std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();
                    const size_t mNew = curr->dataShape[curr->dim - 2];
                    const size_t nNew = curr->dataShape[curr->dim - 1];
                    const size_t kNew =
                        otherShape[otherDim - 1]; // since we want the tranpose of other

                    for (size_t outputVertical = 0; outputVertical < mNew;
                         outputVertical += TensorImpl::tileM) {
                        for (size_t otherHorizontal = 0; otherHorizontal < nNew;
                             otherHorizontal += TensorImpl::tileN) {
                            for (size_t k = 0; k < kNew; k += tileK) {
                                const size_t outputHorizontal = k;
                                const size_t otherVertical = k;
                                const size_t endOutputHorizontal = std::min(k + tileK, kNew);
                                const size_t endOutputVertical =
                                    std::min(outputVertical + TensorImpl::tileM, mNew);
                                const size_t endOtherHorizontal =
                                    std::min(otherHorizontal + TensorImpl::tileN, nNew);

                                for (size_t y = outputVertical; y < endOutputVertical; ++y) {
                                    for (size_t xOther = otherHorizontal;
                                         xOther < endOtherHorizontal;
                                         ++xOther) {
                                        resultType accum = 0;
                                        for (size_t index = 0;
                                             index < endOutputHorizontal - outputHorizontal;
                                             ++index) {
                                            const size_t actualOutputCoordinate =
                                                batchStartIndex + y * outputRowStride +
                                                (outputHorizontal + index) * outputColStride;
                                            const size_t actualOtherCoordinate =
                                                otherCoordinate + xOther * otherRowStride +
                                                (otherVertical + index) *
                                                    otherColStride; // inverted cuz transposed
                                            resultType multipliedResult =
                                                outputGradientVector[actualOutputCoordinate] *
                                                other->data[actualOtherCoordinate];
                                            accum += multipliedResult;
                                        }
                                        const size_t actualCurrCoordinate = currCoordinate +
                                                                            y * currRowStride +
                                                                            xOther * currColStride;
                                        currGradientVector[actualCurrCoordinate] +=
                                            static_cast<scalarType>(accum);
                                    }
                                }
                            }
                        }
                    }
                }

                if (other->trackGradient) {
                    std::vector<otherScalarType>& otherGradientVector =
                        other->ensureGradient().getData();
                    const size_t mNew = otherShape[otherDim - 2];
                    const size_t nNew = otherShape[otherDim - 1];
                    const size_t kNew =
                        curr->dataShape[curr->dim - 2]; // since we want the tranpose of curr

                    for (size_t currVertical = 0; currVertical < mNew;
                         currVertical += TensorImpl::tileM) {
                        for (size_t outputHorizontal = 0; outputHorizontal < nNew;
                             outputHorizontal += TensorImpl::tileN) {
                            for (size_t k = 0; k < kNew; k += tileK) {
                                const size_t currHorizontal = k;
                                const size_t outputVertical = k;
                                const size_t endCurrHorizontal = std::min(k + tileK, kNew);
                                const size_t endCurrVertical =
                                    std::min(currVertical + TensorImpl::tileM, mNew);
                                const size_t endOutputHorizontal =
                                    std::min(outputHorizontal + TensorImpl::tileN, nNew);

                                for (size_t y = currVertical; y < endCurrVertical; ++y) {
                                    for (size_t xCurr = outputHorizontal;
                                         xCurr < endOutputHorizontal;
                                         ++xCurr) {
                                        resultType accum = 0;
                                        for (size_t index = 0;
                                             index < endCurrHorizontal - currHorizontal;
                                             ++index) {
                                            const size_t actualOutputCoordinate =
                                                batchStartIndex +
                                                (outputVertical + index) * outputRowStride +
                                                xCurr * outputColStride;
                                            const size_t actualCurrCoordinate =
                                                currCoordinate +
                                                (currHorizontal + index) * currRowStride +
                                                y * currColStride; // inverted cuz transposed
                                            resultType multipliedResult =
                                                outputGradientVector[actualOutputCoordinate] *
                                                curr->data[actualCurrCoordinate];
                                            accum += multipliedResult;
                                        }
                                        const size_t actualOtherCoordinate = otherCoordinate +
                                                                             y * otherRowStride +
                                                                             xCurr * otherColStride;
                                        otherGradientVector[actualOtherCoordinate] +=
                                            static_cast<otherScalarType>(accum);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        return output;
    }

public:
    std::shared_ptr<TensorImpl<scalarType>> tanh() {
        const size_t totalElements = getNumTotalElements();
        std::vector<scalarType> outputData(totalElements);
        std::vector<size_t> outputShape(dataShape);

        for (size_t outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
            size_t remaining = outputIndex;
            size_t inputCoordinate = 0;

            for (size_t reverseIndex = dim; reverseIndex > 0; --reverseIndex) {
                const size_t dimension = reverseIndex - 1;
                const size_t coordinate = remaining % dataShape[dimension];
                remaining /= dataShape[dimension];
                inputCoordinate += coordinate * stride[dimension];
            }

            outputData[outputIndex] = std::tanh(data[inputCoordinate]);
        }

        std::shared_ptr<TensorImpl<scalarType>> output = std::make_shared<TensorImpl<scalarType>>(
            std::move(outputData), std::move(outputShape), trackGradient);

        if (!trackGradient) {
            return output;
        }

        output->parents = {this->shared_from_this()};

        std::weak_ptr<TensorImpl<scalarType>> currWeak{this->shared_from_this()};
        std::weak_ptr<TensorImpl<scalarType>> outputWeak{output};

        output->backward = [currWeak, outputWeak, totalElements]() {
            std::shared_ptr<TensorImpl<scalarType>> output = outputWeak.lock();
            std::shared_ptr<TensorImpl<scalarType>> curr = currWeak.lock();
            if (!output || !curr || !curr->trackGradient) {
                return;
            }

            std::vector<scalarType>& outputGradientVector = output->ensureGradient().getData();
            std::vector<scalarType>& currentGradientVector = curr->ensureGradient().getData();

            for (size_t outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
                size_t remaining = outputIndex;
                size_t inputCoordinate = 0;

                for (size_t reverseIndex = curr->dim; reverseIndex > 0; --reverseIndex) {
                    const size_t dimension = reverseIndex - 1;
                    const size_t coordinate = remaining % curr->dataShape[dimension];
                    remaining /= curr->dataShape[dimension];
                    inputCoordinate += coordinate * curr->stride[dimension];
                }

                const scalarType outputValue = output->data[outputIndex];
                currentGradientVector[inputCoordinate] +=
                    (scalarType{1} - outputValue * outputValue) * outputGradientVector[outputIndex];
            }
        };

        return output;
    }

    TensorImpl(std::vector<scalarType> inputVector,
               std::vector<size_t> inputDimShape,
               bool shouldTrackGradient)
        : TensorBaseImpl(std::move(inputDimShape)), data(std::move(inputVector)),
          trackGradient(shouldTrackGradient) {
        if constexpr (!isSupportedAutogradScalarType<scalarType>) {
            if (shouldTrackGradient) {
                throw std::runtime_error{"Only floating-point tensors can track gradients"};
            }
        }

        numTotalElements = data.size();
        fillStride(dim - 1);
    }
    bool trackGradient = true;
    std::unique_ptr<TensorImpl<scalarType>> gradient;

    TensorImpl(const TensorImpl& other)
        : TensorBaseImpl(other), data(other.data), trackGradient(other.trackGradient) {
        if (other.gradient) {
            gradient = std::make_unique<TensorImpl<scalarType>>(*other.gradient);
        }
    }

    TensorImpl& operator=(const TensorImpl& other) {
        if (this == &other) {
            return *this;
        }

        TensorBaseImpl::operator=(other);
        data = other.data;
        trackGradient = other.trackGradient;

        if (other.gradient) {
            gradient = std::make_unique<TensorImpl<scalarType>>(*other.gradient);
        } else {
            gradient.reset();
        }

        return *this;
    }

    TensorImpl<scalarType>& ensureGradient() {
        if (!gradient) {
            std::vector<scalarType> zeros(data.size(), scalarType{});
            gradient = std::make_unique<TensorImpl<scalarType>>(std::move(zeros), dataShape, false);
        }

        return *gradient;
    }

    scalarType operator[](size_t index) const {
        return data[index];
    }

    template <typename inputType>
    TensorImpl(const inputType& data, bool shouldTrackGradient = true)
        : trackGradient(shouldTrackGradient) {
        if constexpr (!isSupportedAutogradScalarType<scalarType>) {
            if (shouldTrackGradient) {
                throw std::runtime_error{"Only floating-point tensors can track gradients"};
            }
        }

        findShapeAndFlatten(data, 0);
        numTotalElements = this->data.size();
        stride.resize(dim);
        if (dim > 0) {
            fillStride(dim - 1);
        } else {
            throw std::runtime_error{"dimension must be greater than 0"};
        }
    }
    std::shared_ptr<TensorImpl<scalarType>> transpose(size_t dim1, size_t dim2) {
        if (dim1 >= dim || dim2 >= dim) {
            throw std::out_of_range{"Transpose dimension is out of range"};
        }

        std::shared_ptr<TensorImpl<scalarType>> curr = this->shared_from_this();
        std::shared_ptr<TensorImpl<scalarType>> output =
            std::make_shared<TensorImpl<scalarType>>(data, dataShape, trackGradient);

        // tranposing is equivalent to just switching the coordinates of every element in the matrix
        // so to we just need to swap the stride
        output->stride = stride;
        output->swapStride(dim1, dim2);
        output->swapShape(dim1, dim2);

        if (!trackGradient) {
            return output;
        }

        output->parents = {curr};
        std::weak_ptr<TensorImpl<scalarType>> currWeak{curr};
        std::weak_ptr<TensorImpl<scalarType>> outputWeak{output};

        output->backward = [currWeak, outputWeak]() {
            std::shared_ptr<TensorImpl<scalarType>> curr = currWeak.lock();
            std::shared_ptr<TensorImpl<scalarType>> output = outputWeak.lock();

            if (!curr || !output || !curr->trackGradient) {
                return;
            }

            std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();
            std::vector<scalarType>& outputGradientVector = output->ensureGradient().getData();

            for (size_t index = 0; index < outputGradientVector.size(); ++index) {
                currGradientVector[index] += outputGradientVector[index];
            }
        };

        return output;
    }

    template <typename otherScalarType>
    std::shared_ptr<TensorImpl<std::common_type_t<scalarType, otherScalarType>>>
    operator*(const std::shared_ptr<TensorImpl<otherScalarType>>& other) {
        if (!other) {
            throw std::runtime_error{"Cannot multiply by a null tensor"};
        }

        const size_t otherDim = other->getDim();
        const size_t currDim = dim;

        if (otherDim < 2 || currDim < 2) {
            throw std::runtime_error{"Ensure both tensors are at least 2d"};
        }

        std::vector<size_t> otherShape = other->getShape();
        const size_t mCurr = dataShape[currDim - 2];
        const size_t nCurr = dataShape[currDim - 1];
        const size_t mOther = otherShape[otherDim - 2];
        const size_t nOther = otherShape[otherDim - 1];

        if (mOther != nCurr) {
            throw std::runtime_error("Based on the shapes of the final two dimensions of each "
                                     "matrix, these two tensors cannot be multiplied");
        }

        size_t batchDim;
        size_t totalElements;
        std::vector<size_t> newShape;
        std::vector<size_t> effectiveStrideCurr;
        std::vector<size_t> effectiveStrideOther;
        broadcast(*other,
                  newShape,
                  effectiveStrideCurr,
                  effectiveStrideOther,
                  batchDim,
                  totalElements,
                  2);
        totalElements *= mCurr * nOther;

        newShape.push_back(mCurr);
        newShape.push_back(nOther);

        return tiledMatmul(other,
                           std::move(newShape),
                           effectiveStrideCurr,
                           effectiveStrideOther,
                           totalElements,
                           mCurr,
                           nCurr,
                           nOther,
                           batchDim);
    }

    template <typename otherScalarType>
    std::shared_ptr<TensorImpl<std::common_type_t<scalarType, otherScalarType>>>
    operator+(const std::shared_ptr<TensorImpl<otherScalarType>>& other) {
        if (!other) {
            throw std::runtime_error{"Cannot add with a null tensor"};
        }

        size_t batchDim;
        size_t totalElements;
        std::vector<size_t> newShape;
        std::vector<size_t> effectiveStrideCurr;
        std::vector<size_t> effectiveStrideOther;
        broadcast(*other,
                  newShape,
                  effectiveStrideCurr,
                  effectiveStrideOther,
                  batchDim,
                  totalElements,
                  0);

        using resultType = std::common_type_t<scalarType, otherScalarType>;

        std::vector<resultType> result(totalElements);
        for (size_t outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
            size_t outputIndexCopy = outputIndex;
            size_t currCoordinate = 0;
            size_t otherCoordinate = 0;
            for (size_t reverseIndex = batchDim; reverseIndex > 0; --reverseIndex) {
                const size_t index = reverseIndex - 1;
                const size_t currDimStrideCoefficient = outputIndexCopy % newShape[index];
                outputIndexCopy /= newShape[index];
                currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
            }
            result[outputIndex] = static_cast<resultType>(data[currCoordinate]) +
                                  static_cast<resultType>(other->data[otherCoordinate]);
        }

        bool outputTracksGradient = trackGradient || other->trackGradient;
        std::shared_ptr<TensorImpl<resultType>> output = std::make_shared<TensorImpl<resultType>>(
            std::move(result), std::move(newShape), outputTracksGradient);

        if (outputTracksGradient) {
            output->parents = {this->shared_from_this(), other};
        }
        std::weak_ptr<TensorImpl<scalarType>> currWeak{this->shared_from_this()};
        std::weak_ptr<TensorImpl<otherScalarType>> otherWeak{other};
        std::weak_ptr<TensorImpl<resultType>> outputWeak{output};

        output->backward = [batchDim,
                            currWeak,
                            otherWeak,
                            outputWeak,
                            effectiveStrideCurr = std::move(effectiveStrideCurr),
                            effectiveStrideOther = std::move(effectiveStrideOther),
                            totalElements]() {
            std::shared_ptr<TensorImpl<resultType>> output = outputWeak.lock();
            std::shared_ptr<TensorImpl<scalarType>> curr = currWeak.lock();
            std::shared_ptr<TensorImpl<otherScalarType>> other = otherWeak.lock();

            if (!output || !curr || !other) {
                return;
            }

            std::vector<resultType>& outputGradientVector = output->ensureGradient().getData();
            const std::vector<size_t>& newShape = output->getShape();

            for (size_t outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
                size_t outputIndexCopy = outputIndex;
                size_t currCoordinate = 0;
                size_t otherCoordinate = 0;
                for (size_t reverseIndex = batchDim; reverseIndex > 0; --reverseIndex) {
                    const size_t index = reverseIndex - 1;
                    const size_t currDimStrideCoefficient = outputIndexCopy % newShape[index];
                    outputIndexCopy /= newShape[index];
                    currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                    otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
                }
                if (other->trackGradient) {
                    std::vector<otherScalarType>& otherGradientVector =
                        other->ensureGradient().getData();
                    otherGradientVector[otherCoordinate] +=
                        static_cast<otherScalarType>(outputGradientVector[outputIndex]);
                }

                if (curr->trackGradient) {
                    std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();
                    currGradientVector[currCoordinate] +=
                        static_cast<scalarType>(outputGradientVector[outputIndex]);
                }
            }
        };

        return output;
    }

    template <typename otherScalarType,
              std::enable_if_t<std::is_arithmetic_v<otherScalarType>, int> = 0>
    std::shared_ptr<TensorImpl<std::common_type_t<scalarType, otherScalarType>>>
    operator+(otherScalarType other) {
        using resultType = std::common_type_t<scalarType, otherScalarType>;

        const size_t totalElements = data.size();
        std::vector<resultType> result(totalElements);

        for (size_t outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
            size_t remaining = outputIndex;
            size_t currCoordinate = 0;

            for (size_t reverseIndex = dim; reverseIndex > 0; --reverseIndex) {
                const size_t index = reverseIndex - 1;
                const size_t coordinate = remaining % dataShape[index];
                remaining /= dataShape[index];
                currCoordinate += coordinate * stride[index];
            }

            result[outputIndex] =
                static_cast<resultType>(data[currCoordinate]) + static_cast<resultType>(other);
        }

        std::shared_ptr<TensorImpl<resultType>> output =
            std::make_shared<TensorImpl<resultType>>(std::move(result), dataShape, trackGradient);

        if (trackGradient) {
            output->parents = {this->shared_from_this()};
        }

        std::weak_ptr<TensorImpl<scalarType>> currWeak{this->shared_from_this()};
        std::weak_ptr<TensorImpl<resultType>> outputWeak{output};

        output->backward = [currWeak, outputWeak, totalElements]() {
            std::shared_ptr<TensorImpl<scalarType>> curr = currWeak.lock();
            std::shared_ptr<TensorImpl<resultType>> output = outputWeak.lock();

            if (!curr || !output || !curr->trackGradient) {
                return;
            }

            std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();
            std::vector<resultType>& outputGradientVector = output->ensureGradient().getData();
            const std::vector<size_t>& currShape = curr->getShape();
            const std::vector<size_t>& currStride = curr->getStride();

            for (size_t outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
                size_t remaining = outputIndex;
                size_t currCoordinate = 0;

                for (size_t reverseIndex = currShape.size(); reverseIndex > 0; --reverseIndex) {
                    const size_t index = reverseIndex - 1;
                    const size_t coordinate = remaining % currShape[index];
                    remaining /= currShape[index];
                    currCoordinate += coordinate * currStride[index];
                }

                currGradientVector[currCoordinate] +=
                    static_cast<scalarType>(outputGradientVector[outputIndex]);
            }
        };

        return output;
    }

    template <typename otherScalarType,
              std::enable_if_t<std::is_arithmetic_v<otherScalarType>, int> = 0>
    std::shared_ptr<TensorImpl<std::common_type_t<scalarType, otherScalarType>>>
    operator*(otherScalarType other) {
        using resultType = std::common_type_t<scalarType, otherScalarType>;

        const size_t totalElements = data.size();
        const resultType scalar = static_cast<resultType>(other);
        std::vector<resultType> result(totalElements);

        for (size_t outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
            size_t remaining = outputIndex;
            size_t currCoordinate = 0;

            for (size_t reverseIndex = dim; reverseIndex > 0; --reverseIndex) {
                const size_t index = reverseIndex - 1;
                const size_t coordinate = remaining % dataShape[index];
                remaining /= dataShape[index];
                currCoordinate += coordinate * stride[index];
            }

            result[outputIndex] = static_cast<resultType>(data[currCoordinate]) * scalar;
        }

        std::shared_ptr<TensorImpl<resultType>> output =
            std::make_shared<TensorImpl<resultType>>(std::move(result), dataShape, trackGradient);

        if (trackGradient) {
            output->parents = {this->shared_from_this()};
        }

        std::weak_ptr<TensorImpl<scalarType>> currWeak{this->shared_from_this()};
        std::weak_ptr<TensorImpl<resultType>> outputWeak{output};

        output->backward = [currWeak, outputWeak, totalElements, scalar]() {
            std::shared_ptr<TensorImpl<scalarType>> curr = currWeak.lock();
            std::shared_ptr<TensorImpl<resultType>> output = outputWeak.lock();

            if (!curr || !output || !curr->trackGradient) {
                return;
            }

            std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();
            std::vector<resultType>& outputGradientVector = output->ensureGradient().getData();
            const std::vector<size_t>& currShape = curr->getShape();
            const std::vector<size_t>& currStride = curr->getStride();

            for (size_t outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
                size_t remaining = outputIndex;
                size_t currCoordinate = 0;

                for (size_t reverseIndex = currShape.size(); reverseIndex > 0; --reverseIndex) {
                    const size_t index = reverseIndex - 1;
                    const size_t coordinate = remaining % currShape[index];
                    remaining /= currShape[index];
                    currCoordinate += coordinate * currStride[index];
                }

                currGradientVector[currCoordinate] +=
                    static_cast<scalarType>(outputGradientVector[outputIndex] * scalar);
            }
        };

        return output;
    }

    std::vector<scalarType> getDataCopy() {
        return data;
    }

    std::shared_ptr<TensorImpl<scalarType>> softmax() {
        static_assert(isSupportedFloatingPointScalarType<scalarType>,
                      "Softmax requires a floating point tensor type");

        if (dim == 0 || dataShape[dim - 1] == 0) {
            throw std::runtime_error{"Softmax requires a non empty final dimension"};
        }

        std::vector<scalarType> outputData(getNumTotalElements());
        std::vector<size_t> outputShape(dataShape);

        const size_t numElementsPerBatch = dataShape[dim - 1];
        const size_t totalBatches = getNumTotalElements() / numElementsPerBatch;
        const size_t classStride = stride[dim - 1];

        for (size_t batchIndex = 0; batchIndex < totalBatches; ++batchIndex) {
            size_t remaining = batchIndex;
            size_t inputBase = 0;

            for (size_t reverseIndex = dim - 1; reverseIndex > 0; --reverseIndex) {
                size_t dimension = reverseIndex - 1;
                size_t coordinate = remaining % dataShape[dimension];
                remaining /= dataShape[dimension];
                inputBase += coordinate * stride[dimension];
            }

            size_t outputBase = batchIndex * numElementsPerBatch;
            scalarType softmaxAccum = 0;
            scalarType largestLogit = data[inputBase];

            for (size_t index = 1; index < numElementsPerBatch; ++index) {
                largestLogit = std::max(largestLogit, data[inputBase + index * classStride]);
            }

            for (size_t index = 0; index < numElementsPerBatch; ++index) {
                softmaxAccum += std::exp(data[inputBase + index * classStride] - largestLogit);
            }

            for (size_t index = 0; index < numElementsPerBatch; ++index) {
                outputData[outputBase + index] =
                    std::exp(data[inputBase + index * classStride] - largestLogit) / softmaxAccum;
            }
        }

        std::shared_ptr<TensorImpl<scalarType>> output = std::make_shared<TensorImpl<scalarType>>(
            std::move(outputData), std::move(outputShape), trackGradient);

        if (!trackGradient) {
            return output;
        }

        std::shared_ptr<TensorImpl<scalarType>> curr = this->shared_from_this();
        output->parents = {curr};
        std::weak_ptr<TensorImpl<scalarType>> currWeak{curr};
        std::weak_ptr<TensorImpl<scalarType>> outputWeak{output};

        output->backward = [currWeak, outputWeak, numElementsPerBatch, totalBatches]() {
            std::shared_ptr<TensorImpl<scalarType>> curr = currWeak.lock();
            std::shared_ptr<TensorImpl<scalarType>> output = outputWeak.lock();

            if (!curr || !output || !curr->trackGradient) {
                return;
            }

            std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();
            std::vector<scalarType>& outputGradientVector = output->ensureGradient().getData();
            const size_t inputClassStride = curr->stride[curr->dim - 1];

            for (size_t batchIndex = 0; batchIndex < totalBatches; ++batchIndex) {
                size_t remaining = batchIndex;
                size_t inputBase = 0;

                for (size_t reverseIndex = curr->dim - 1; reverseIndex > 0; --reverseIndex) {
                    const size_t dimension = reverseIndex - 1;
                    const size_t coordinate = remaining % curr->dataShape[dimension];
                    remaining /= curr->dataShape[dimension];
                    inputBase += coordinate * curr->stride[dimension];
                }

                const size_t outputBase = batchIndex * numElementsPerBatch;
                scalarType gradientDotOutput = 0;

                for (size_t index = 0; index < numElementsPerBatch; ++index) {
                    gradientDotOutput +=
                        outputGradientVector[outputBase + index] * output->data[outputBase + index];
                }

                for (size_t index = 0; index < numElementsPerBatch; ++index) {
                    currGradientVector[inputBase + index * inputClassStride] +=
                        output->data[outputBase + index] *
                        (outputGradientVector[outputBase + index] - gradientDotOutput);
                }
            }
        };

        return output;
    }

    void step(scalarType learningRate) {
        if (!gradient) {
            return;
        }
        std::vector<scalarType>& gradientData = gradient->getData();
        for (size_t i = 0; i < getNumTotalElements(); ++i) {
            data[i] -= gradientData[i] * learningRate;
        }
    }

    void zeroGrad() {
        if (!gradient) {
            return;
        }
        std::vector<scalarType>& gradientData = gradient->getData();
        std::fill(gradientData.begin(), gradientData.end(), scalarType{});
    }

    template <typename anyType>
    std::shared_ptr<TensorImpl<scalarType>> NLLLoss(std::shared_ptr<TensorImpl<anyType>>& target) {
        static_assert(isSupportedFloatingPointScalarType<scalarType>,
                      "NLLLoss predictions require a supported floating-point scalar type");
        static_assert(isSupportedIndexScalarType<anyType>,
                      "NLLLoss targets require a supported integer scalar type");

        const size_t numElementsPerBatch = dataShape[dim - 1];
        const size_t totalBatches = getNumTotalElements() / numElementsPerBatch;
        std::vector<size_t> selectedIndices(totalBatches);
        std::vector<scalarType> outputData(1);
        std::vector<size_t> outputShape{1};

        scalarType totalLoss = 0;
        for (size_t batchIndex = 0; batchIndex < totalBatches; batchIndex += 1) {
            size_t targetClass = target->data[batchIndex];
            size_t selectedIndex = batchIndex * numElementsPerBatch + targetClass;
            selectedIndices[batchIndex] = selectedIndex;
            scalarType probability =
                std::max(data[selectedIndex], std::numeric_limits<scalarType>::min());
            totalLoss += -std::log(probability);
        }

        outputData[0] = totalLoss / totalBatches;

        std::shared_ptr<TensorImpl<scalarType>> output = std::make_shared<TensorImpl<scalarType>>(
            std::move(outputData), std::move(outputShape), trackGradient);

        std::shared_ptr<TensorImpl<scalarType>> curr = this->shared_from_this();
        output->parents = {curr};

        std::weak_ptr<TensorImpl<scalarType>> outputWeak{output};
        std::weak_ptr<TensorImpl<scalarType>> currWeak{curr};

        output->backward =
            [outputWeak, currWeak, totalBatches, selectedIndices = std::move(selectedIndices)]() {
                std::shared_ptr<TensorImpl<scalarType>> output = outputWeak.lock();
                std::shared_ptr<TensorImpl<scalarType>> curr = currWeak.lock();

                if (!output || !curr || !curr->trackGradient) {
                    return;
                }

                std::vector<scalarType>& outputGradientVector = output->ensureGradient().getData();
                std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();

                for (size_t batchIndex = 0; batchIndex < selectedIndices.size(); batchIndex += 1) {
                    size_t selectedIndex = selectedIndices[batchIndex];
                    scalarType probability =
                        std::max(curr->data[selectedIndex], std::numeric_limits<scalarType>::min());

                    currGradientVector[selectedIndex] +=
                        outputGradientVector[0] * (-1 / probability) / totalBatches;
                }
            };

        return output;
    }

    void backwardPass() {
        if (getNumTotalElements() != 1) {
            throw std::runtime_error{"backward() requires a scalar tensor"};
        }

        std::vector<scalarType>& rootGradient = ensureGradient().getData();

        rootGradient[0] = scalarType{1};
        applyBackward();
    }
};
