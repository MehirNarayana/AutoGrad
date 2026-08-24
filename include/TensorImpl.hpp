#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

template <typename scalarType> class TensorImpl;

class TensorBaseImpl {
protected:
    size_t dim = 0;
    std::vector<size_t> dataShape;
    std::vector<size_t> stride;

    void fillStride(size_t lastIndex);

    void swapStride(size_t dim1, size_t dim2);
    void swapShape(size_t dim1, size_t dim2);
    TensorBaseImpl();
    TensorBaseImpl(std::vector<size_t> dataShape);

public:
    virtual ~TensorBaseImpl() = default;
    size_t getDim() const noexcept;
    const std::vector<size_t>& getShape() const noexcept;
    const std::vector<size_t>& getStride() const noexcept;
    std::function<void()> backward;
    int getNumTotalElements() const;
};

template <typename scalarType = float>
class TensorImpl : public TensorBaseImpl,
                   public std::enable_shared_from_this<TensorImpl<scalarType>> {
private:
    template <typename> friend class TensorImpl;

    std::vector<scalarType> data;
    std::vector<std::shared_ptr<TensorBaseImpl>> parents;

    const static int tileM = 32;
    const static int tileN = 32;
    const static int tileK = 32;

    std::vector<scalarType>& getData() {
        return data;
    }

    template <typename inputType> struct isVector : std::false_type {};

    template <typename inputType, typename Alloc>
    struct isVector<std::vector<inputType, Alloc>> : std::true_type {};

    bool isScalarType(scalarType x) {
        return true;
    }

    template <typename randomType> bool isScalarType(randomType x) {
        return false;
    }

    template <typename inputType> size_t findShapeAndFlatten(const inputType& data, int currDepth) {
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
                   int& batchDim,
                   int& totalElements,
                   int ignoredDimensions) {
        int otherDim = static_cast<int>(other.getDim());
        int currDim = static_cast<int>(dim);

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
        int currDimIndex = currDim - ignoredDimensions - 1;
        int otherDimIndex = otherDim - ignoredDimensions - 1;
        int k = batchDim - 1;

        while (currDimIndex >= 0 && otherDimIndex >= 0) {
            size_t a = dataShape[currDimIndex];
            size_t b = otherShape[otherDimIndex];

            if (a == b) {
                newShape[k] = a;
                effectiveStrideCurr[k] = stride[currDimIndex];
                effectiveStrideOther[k] = otherStride[otherDimIndex];
            } else if (a == 1) {
                newShape[k] = b;
                effectiveStrideCurr[k] = 0;
                effectiveStrideOther[k] = otherStride[otherDimIndex];
            } else if (b == 1) {
                newShape[k] = a;
                effectiveStrideCurr[k] = stride[currDimIndex];
                effectiveStrideOther[k] = 0;
            } else {
                throw std::runtime_error("Tensors cannot be broadcast together");
            }

            totalElements *= newShape[k];
            --k;
            --currDimIndex;
            --otherDimIndex;
        }

        while (currDimIndex >= 0) {
            newShape[k] = dataShape[currDimIndex];
            effectiveStrideCurr[k] = stride[currDimIndex];
            effectiveStrideOther[k] = 0;
            totalElements *= newShape[k];
            --k;
            --currDimIndex;
        }

        while (otherDimIndex >= 0) {
            newShape[k] = otherShape[otherDimIndex];
            effectiveStrideCurr[k] = 0;
            effectiveStrideOther[k] = otherStride[otherDimIndex];
            totalElements *= newShape[k];
            --k;
            --otherDimIndex;
        }
    }

    template <typename otherScalarType>
    std::shared_ptr<TensorImpl<std::common_type_t<scalarType, otherScalarType>>>
    tiledMatmul(const std::shared_ptr<TensorImpl<otherScalarType>>& other,
                std::vector<size_t> newShape,
                const std::vector<size_t>& effectiveStrideCurr,
                const std::vector<size_t>& effectiveStrideOther,
                int totalElements,
                int mCurr,
                int nCurr,
                int nOther,
                int batchDim) {
        int elementsPerBatch = mCurr * nOther;
        int totalBatches = totalElements / elementsPerBatch;

        using resultType = std::common_type_t<scalarType, otherScalarType>;

        std::vector<resultType> result(totalElements);
        size_t currRowStride = stride[dim - 2];
        size_t currColStride = stride[dim - 1];

        int otherDim = static_cast<int>(other->getDim());
        const std::vector<size_t>& otherStride = other->getStride();
        size_t otherRowStride = otherStride[otherDim - 2];
        size_t otherColStride = otherStride[otherDim - 1];

        for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
            int batchIndexCopy = batchIndex;
            int currCoordinate = 0;
            int otherCoordinate = 0;
            for (int index = batchDim - 1; index >= 0; index--) {
                int currDimStrideCoefficient = batchIndexCopy % newShape[index];
                batchIndexCopy /= newShape[index];
                currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
            }

            int batchStartIndex = batchIndex * elementsPerBatch;
            int mNew = newShape[batchDim];
            int nNew = newShape[batchDim + 1];
            int kNew = nCurr;
            for (int currVertical = 0; currVertical < mNew; currVertical += TensorImpl::tileM) {
                for (int otherHorizontal = 0; otherHorizontal < nNew;
                     otherHorizontal += TensorImpl::tileN) {
                    for (int k = 0; k < kNew; k += tileK) {
                        int currHorizontal = k;
                        int otherVertical = k;
                        int endCurrHorizontal = std::min(k + tileK, kNew);
                        int endCurrVertical = std::min(currVertical + TensorImpl::tileM, mNew);
                        int endOtherHorizontal =
                            std::min(otherHorizontal + TensorImpl::tileN, nNew);

                        for (int y = currVertical; y < endCurrVertical; y++) {
                            for (int xOther = otherHorizontal; xOther < endOtherHorizontal;
                                 xOther++) {
                                resultType accum = 0;
                                for (int index = 0; index < endCurrHorizontal - currHorizontal;
                                     index++) {
                                    int actualCurrCoordinate =
                                        currCoordinate + y * currRowStride +
                                        (currHorizontal + index) * currColStride;
                                    int actualOtherCoordinate =
                                        otherCoordinate + (otherVertical + index) * otherRowStride +
                                        xOther * otherColStride;
                                    resultType multipliedResult =
                                        static_cast<resultType>(data[actualCurrCoordinate]) *
                                        static_cast<resultType>((*other)[actualOtherCoordinate]);
                                    accum += multipliedResult;
                                }
                                int targetCoordinate = batchStartIndex + y * nNew + xOther;
                                result[targetCoordinate] += accum;
                            }
                        }
                    }
                }
            }
        }

        bool outputTracksGradient = trackGradient || other->trackGradient;
        std::shared_ptr<TensorImpl<resultType>> output(new TensorImpl<resultType>(
            std::move(result), std::move(newShape), outputTracksGradient));
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
            int outputDim = static_cast<int>(output->getDim());
            size_t outputRowStride = outputStride[outputDim - 2];
            size_t outputColStride = outputStride[outputDim - 1];

            size_t currRowStride = curr->stride[curr->dim - 2];
            size_t currColStride = curr->stride[curr->dim - 1];

            const std::vector<size_t>& otherShape = other->getShape();
            const std::vector<size_t>& otherStride = other->getStride();
            int otherDim = static_cast<int>(other->getDim());
            size_t otherRowStride = otherStride[otherDim - 2];
            size_t otherColStride = otherStride[otherDim - 1];

            for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
                int batchIndexCopy = batchIndex;
                int currCoordinate = 0;
                int otherCoordinate = 0;

                for (int index = batchDim - 1; index >= 0; index--) {
                    int currDimStrideCoefficient = batchIndexCopy % outputShape[index];
                    batchIndexCopy /= outputShape[index];
                    currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                    otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
                }

                int batchStartIndex = batchIndex * elementsPerBatch;

                if (curr->trackGradient) {
                    std::vector<scalarType>& currGradientVector = curr->ensureGradient().getData();
                    int mNew = curr->dataShape[curr->dim - 2];
                    int nNew = curr->dataShape[curr->dim - 1];
                    int kNew = otherShape[otherDim - 1]; // since we want the tranpose of other

                    for (int outputVertical = 0; outputVertical < mNew;
                         outputVertical += TensorImpl::tileM) {
                        for (int otherHorizontal = 0; otherHorizontal < nNew;
                             otherHorizontal += TensorImpl::tileN) {
                            for (int k = 0; k < kNew; k += tileK) {
                                int outputHorizontal = k;
                                int otherVertical = k;
                                int endOutputHorizontal = std::min(k + tileK, kNew);
                                int endOutputVertical =
                                    std::min(outputVertical + TensorImpl::tileM, mNew);
                                int endOtherHorizontal =
                                    std::min(otherHorizontal + TensorImpl::tileN, nNew);

                                for (int y = outputVertical; y < endOutputVertical; y++) {
                                    for (int xOther = otherHorizontal; xOther < endOtherHorizontal;
                                         xOther++) {
                                        resultType accum = 0;
                                        for (int index = 0;
                                             index < endOutputHorizontal - outputHorizontal;
                                             index++) {
                                            int actualOutputCoordinate =
                                                batchStartIndex + y * outputRowStride +
                                                (outputHorizontal + index) * outputColStride;
                                            int actualOtherCoordinate =
                                                otherCoordinate + xOther * otherRowStride +
                                                (otherVertical + index) *
                                                    otherColStride; // inverted cuz transposed
                                            resultType multipliedResult =
                                                outputGradientVector[actualOutputCoordinate] *
                                                other->data[actualOtherCoordinate];
                                            accum += multipliedResult;
                                        }
                                        int actualCurrCoordinate = currCoordinate +
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
                    int mNew = otherShape[otherDim - 2];
                    int nNew = otherShape[otherDim - 1];
                    int kNew = curr->dataShape[curr->dim - 2]; // since we want the tranpose of curr

                    for (int currVertical = 0; currVertical < mNew;
                         currVertical += TensorImpl::tileM) {
                        for (int outputHorizontal = 0; outputHorizontal < nNew;
                             outputHorizontal += TensorImpl::tileN) {
                            for (int k = 0; k < kNew; k += tileK) {
                                int currHorizontal = k;
                                int outputVertical = k;
                                int endCurrHorizontal = std::min(k + tileK, kNew);
                                int endCurrVertical =
                                    std::min(currVertical + TensorImpl::tileM, mNew);
                                int endOutputHorizontal =
                                    std::min(outputHorizontal + TensorImpl::tileN, nNew);

                                for (int y = currVertical; y < endCurrVertical; y++) {
                                    for (int xCurr = outputHorizontal; xCurr < endOutputHorizontal;
                                         xCurr++) {
                                        resultType accum = 0;
                                        for (int index = 0;
                                             index < endCurrHorizontal - currHorizontal;
                                             index++) {
                                            int actualOutputCoordinate =
                                                batchStartIndex +
                                                (outputVertical + index) * outputRowStride +
                                                xCurr * outputColStride;
                                            int actualCurrCoordinate =
                                                currCoordinate +
                                                (currHorizontal + index) * currRowStride +
                                                y * currColStride; // inverted cuz transposed
                                            resultType multipliedResult =
                                                outputGradientVector[actualOutputCoordinate] *
                                                curr->data[actualCurrCoordinate];
                                            accum += multipliedResult;
                                        }
                                        int actualOtherCoordinate = otherCoordinate +
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
    TensorImpl(std::vector<scalarType> inputVector,
               std::vector<size_t> inputDimShape,
               bool shouldTrackGradient)
        : TensorBaseImpl(std::move(inputDimShape)), data(std::move(inputVector)),
          trackGradient(shouldTrackGradient) {
        fillStride(dim - 1);
    }
    bool trackGradient = true;
    std::unique_ptr<TensorImpl<scalarType>> gradient;

    TensorImpl(const TensorImpl& other)
        : TensorBaseImpl(other), data(other.data), parents(other.parents),
          trackGradient(other.trackGradient) {
        if (other.gradient) {
            gradient = std::unique_ptr<TensorImpl<scalarType>>(
                new TensorImpl<scalarType>(*other.gradient));
        }
    }

    TensorImpl& operator=(const TensorImpl& other) {
        if (this == &other) {
            return *this;
        }

        TensorBaseImpl::operator=(other);
        data = other.data;
        parents = other.parents;
        trackGradient = other.trackGradient;

        if (other.gradient) {
            gradient = std::unique_ptr<TensorImpl<scalarType>>(
                new TensorImpl<scalarType>(*other.gradient));
        } else {
            gradient.reset();
        }

        return *this;
    }

    TensorImpl<scalarType>& ensureGradient() {
        if (!gradient) {
            std::vector<scalarType> zeros(data.size(), scalarType{});
            gradient = std::unique_ptr<TensorImpl<scalarType>>(
                new TensorImpl<scalarType>(std::move(zeros), dataShape, false));
        }

        return *gradient;
    }

    scalarType operator[](size_t index) const {
        return data[index];
    }

    template <typename inputType>
    TensorImpl(const inputType& data, bool shouldTrackGradient = true)
        : trackGradient(shouldTrackGradient) {
        findShapeAndFlatten(data, 0);
        stride.resize(dim);
        if (dim > 0) {
            fillStride(dim - 1);
        } else {
            throw std::runtime_error{"dimension must be greater than 0"};
        }
    }
    TensorImpl transpose(size_t dim1, size_t dim2) {
        TensorImpl<scalarType> output = *this;
        // tranposing is equivalent to just switching the coordinates of every element in the matrix
        // so to we just need to swap the stride
        output.swapStride(dim1, dim2);
        output.swapShape(dim1, dim2);
        return output;
    }

    template <typename otherScalarType>
    std::shared_ptr<TensorImpl<std::common_type_t<scalarType, otherScalarType>>>
    operator*(const std::shared_ptr<TensorImpl<otherScalarType>>& other) {
        if (!other) {
            throw std::runtime_error{"Cannot multiply by a null tensor"};
        }

        int otherDim = static_cast<int>(other->getDim());
        int currDim = static_cast<int>(dim);

        if (otherDim < 2 || currDim < 2) {
            throw std::runtime_error{"Ensure both tensors are at least 2d"};
        }

        std::vector<size_t> otherShape = other->getShape();
        int mCurr = dataShape[currDim - 2];
        int nCurr = dataShape[currDim - 1];
        int mOther = otherShape[otherDim - 2];
        int nOther = otherShape[otherDim - 1];

        if (mOther != nCurr) {
            throw std::runtime_error("Based on the shapes of the final two dimensions of each "
                                     "matrix, these two tensors cannot be multiplied");
        }

        int batchDim;
        int totalElements;
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

        int batchDim;
        int totalElements;
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
        for (int outputIndex = 0; outputIndex < totalElements; outputIndex++) {
            int outputIndexCopy = outputIndex;
            int currCoordinate = 0;
            int otherCoordinate = 0;
            for (int index = batchDim - 1; index >= 0; index--) {
                int currDimStrideCoefficient = outputIndexCopy % newShape[index];
                outputIndexCopy /= newShape[index];
                currCoordinate += currDimStrideCoefficient * effectiveStrideCurr[index];
                otherCoordinate += currDimStrideCoefficient * effectiveStrideOther[index];
            }
            result[outputIndex] = static_cast<resultType>(data[currCoordinate]) +
                                  static_cast<resultType>(other->data[otherCoordinate]);
        }

        bool outputTracksGradient = trackGradient || other->trackGradient;
        std::shared_ptr<TensorImpl<resultType>> output(new TensorImpl<resultType>(
            std::move(result), std::move(newShape), outputTracksGradient));

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

            for (int outputIndex = 0; outputIndex < totalElements; outputIndex++) {
                int outputIndexCopy = outputIndex;
                int currCoordinate = 0;
                int otherCoordinate = 0;
                for (int index = batchDim - 1; index >= 0; index--) {
                    int currDimStrideCoefficient = outputIndexCopy % newShape[index];
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

        const int totalElements = static_cast<int>(data.size());
        std::vector<resultType> result(totalElements);

        for (int outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
            int remaining = outputIndex;
            int currCoordinate = 0;

            for (int index = static_cast<int>(dim) - 1; index >= 0; --index) {
                int coordinate = remaining % dataShape[index];
                remaining /= dataShape[index];
                currCoordinate += coordinate * stride[index];
            }

            result[outputIndex] =
                static_cast<resultType>(data[currCoordinate]) + static_cast<resultType>(other);
        }

        std::shared_ptr<TensorImpl<resultType>> output(
            new TensorImpl<resultType>(std::move(result), dataShape, trackGradient));

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

            for (int outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
                int remaining = outputIndex;
                int currCoordinate = 0;

                for (int index = static_cast<int>(currShape.size()) - 1; index >= 0; --index) {
                    int coordinate = remaining % currShape[index];
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

        const int totalElements = static_cast<int>(data.size());
        const resultType scalar = static_cast<resultType>(other);
        std::vector<resultType> result(totalElements);

        for (int outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
            int remaining = outputIndex;
            int currCoordinate = 0;

            for (int index = static_cast<int>(dim) - 1; index >= 0; --index) {
                int coordinate = remaining % dataShape[index];
                remaining /= dataShape[index];
                currCoordinate += coordinate * stride[index];
            }

            result[outputIndex] = static_cast<resultType>(data[currCoordinate]) * scalar;
        }

        std::shared_ptr<TensorImpl<resultType>> output(
            new TensorImpl<resultType>(std::move(result), dataShape, trackGradient));

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

            for (int outputIndex = 0; outputIndex < totalElements; ++outputIndex) {
                int remaining = outputIndex;
                int currCoordinate = 0;

                for (int index = static_cast<int>(currShape.size()) - 1; index >= 0; --index) {
                    int coordinate = remaining % currShape[index];
                    remaining /= currShape[index];
                    currCoordinate += coordinate * currStride[index];
                }

                currGradientVector[currCoordinate] +=
                    static_cast<scalarType>(outputGradientVector[outputIndex] * scalar);
            }
        };

        return output;
    }
};
