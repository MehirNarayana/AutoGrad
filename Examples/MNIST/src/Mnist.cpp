#include <Layers/Activations.hpp>
#include <Layers/Linear.hpp>
#include <Layers/LossFns/CrossEntropy.hpp>
#include <Layers/Sequential.hpp>
#include <Optimizer.hpp>
#include <Tensor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef AUTOGRAD_MNIST_DATA_DIRECTORY
#define AUTOGRAD_MNIST_DATA_DIRECTORY "Examples/MNIST/dataset"
#endif

namespace {

constexpr size_t numberOfClasses = 10;
constexpr size_t hiddenSize = 128;

struct MnistDataset {
    std::vector<std::uint8_t> images;
    std::vector<std::uint8_t> labels;
    size_t rows = 0;
    size_t columns = 0;

    size_t size() const noexcept {
        return labels.size();
    }

    size_t pixelsPerImage() const noexcept {
        return rows * columns;
    }
};

struct Batch {
    Tensor<float> images;
    Tensor<size_t> labels;
};

std::uint32_t readBigEndianUint32(std::ifstream& stream) {
    std::array<std::uint8_t, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error{"Unexpected end of an MNIST IDX file"};
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) | static_cast<std::uint32_t>(bytes[3]);
}

MnistDataset loadMnistDataset(const std::filesystem::path& imagePath,
                              const std::filesystem::path& labelPath) {
    std::ifstream imageStream(imagePath, std::ios::binary);
    std::ifstream labelStream(labelPath, std::ios::binary);
    if (!imageStream) {
        throw std::runtime_error{"Could not open image file: " + imagePath.string()};
    }
    if (!labelStream) {
        throw std::runtime_error{"Could not open label file: " + labelPath.string()};
    }

    constexpr std::uint32_t imageMagicNumber = 2051;
    constexpr std::uint32_t labelMagicNumber = 2049;
    if (readBigEndianUint32(imageStream) != imageMagicNumber) {
        throw std::runtime_error{"Invalid MNIST image file: " + imagePath.string()};
    }
    if (readBigEndianUint32(labelStream) != labelMagicNumber) {
        throw std::runtime_error{"Invalid MNIST label file: " + labelPath.string()};
    }

    const size_t imageCount = readBigEndianUint32(imageStream);
    const size_t labelCount = readBigEndianUint32(labelStream);
    const size_t rows = readBigEndianUint32(imageStream);
    const size_t columns = readBigEndianUint32(imageStream);

    if (imageCount != labelCount) {
        throw std::runtime_error{"MNIST image and label counts do not match"};
    }
    if (imageCount == 0 || rows == 0 || columns == 0) {
        throw std::runtime_error{"MNIST dataset dimensions must be nonzero"};
    }
    if (rows * columns != 784) {
        throw std::runtime_error{"This MLP expects 28x28 MNIST images"};
    }

    MnistDataset dataset;
    dataset.rows = rows;
    dataset.columns = columns;
    dataset.images.resize(imageCount * rows * columns);
    dataset.labels.resize(labelCount);

    imageStream.read(reinterpret_cast<char*>(dataset.images.data()),
                     static_cast<std::streamsize>(dataset.images.size()));
    labelStream.read(reinterpret_cast<char*>(dataset.labels.data()),
                     static_cast<std::streamsize>(dataset.labels.size()));
    if (!imageStream || !labelStream) {
        throw std::runtime_error{"An MNIST IDX file is truncated"};
    }

    for (std::uint8_t label : dataset.labels) {
        if (label >= numberOfClasses) {
            throw std::runtime_error{"MNIST label is outside the range [0, 9]"};
        }
    }

    return dataset;
}

Batch makeBatch(const MnistDataset& dataset,
                const std::vector<size_t>& order,
                size_t batchStart,
                size_t requestedBatchSize) {
    const size_t batchSize = std::min(requestedBatchSize, dataset.size() - batchStart);
    const size_t pixelsPerImage = dataset.pixelsPerImage();
    std::vector<float> imageData(batchSize * pixelsPerImage);
    std::vector<size_t> labelData(batchSize);

    for (size_t batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        const size_t datasetIndex = order[batchStart + batchIndex];
        const size_t sourceBase = datasetIndex * pixelsPerImage;
        const size_t destinationBase = batchIndex * pixelsPerImage;

        for (size_t pixel = 0; pixel < pixelsPerImage; ++pixel) {
            imageData[destinationBase + pixel] =
                static_cast<float>(dataset.images[sourceBase + pixel]) / 255.0F;
        }
        labelData[batchIndex] = dataset.labels[datasetIndex];
    }

    return {
        Tensor<float>(std::move(imageData), {batchSize, pixelsPerImage}, false),
        Tensor<size_t>(std::move(labelData), {batchSize}, false),
    };
}

size_t countCorrectPredictions(const Tensor<float>& logits,
                               const MnistDataset& dataset,
                               const std::vector<size_t>& order,
                               size_t batchStart,
                               size_t batchSize) {
    size_t correct = 0;
    for (size_t batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        const size_t outputBase = batchIndex * numberOfClasses;
        size_t predictedClass = 0;
        for (size_t classIndex = 1; classIndex < numberOfClasses; ++classIndex) {
            if (logits[outputBase + classIndex] > logits[outputBase + predictedClass]) {
                predictedClass = classIndex;
            }
        }

        if (predictedClass == dataset.labels[order[batchStart + batchIndex]]) {
            ++correct;
        }
    }
    return correct;
}

float evaluate(Sequential<float>& model, const MnistDataset& dataset, size_t batchSize) {
    std::vector<size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), size_t{0});
    size_t correct = 0;

    for (size_t batchStart = 0; batchStart < dataset.size(); batchStart += batchSize) {
        const size_t currentBatchSize = std::min(batchSize, dataset.size() - batchStart);
        Batch batch = makeBatch(dataset, order, batchStart, batchSize);
        Tensor<float> logits = model.forward(std::move(batch.images));
        correct += countCorrectPredictions(logits, dataset, order, batchStart, currentBatchSize);
    }

    return static_cast<float>(correct) / static_cast<float>(dataset.size());
}

void printUsage(const char* executable) {
    std::cout << "Usage: " << executable
              << " [mnist-directory] [epochs] [batch-size] [learning-rate]\n\n"
              << "Default dataset directory:\n"
              << "  " << AUTOGRAD_MNIST_DATA_DIRECTORY << "\n\n"
              << "The directory must contain the extracted IDX files:\n"
              << "  train-images-idx3-ubyte\n"
              << "  train-labels-idx1-ubyte\n"
              << "  t10k-images-idx3-ubyte\n"
              << "  t10k-labels-idx1-ubyte\n";
}

} // namespace

int main(int argumentCount, char** arguments) {
    if (argumentCount > 1 && std::string(arguments[1]) == "--help") {
        printUsage(arguments[0]);
        return 0;
    }

    try {
        const std::filesystem::path datasetDirectory =
            argumentCount > 1 ? arguments[1] : AUTOGRAD_MNIST_DATA_DIRECTORY;
        const size_t epochs = argumentCount > 2 ? std::stoull(arguments[2]) : 5;
        const size_t batchSize = argumentCount > 3 ? std::stoull(arguments[3]) : 64;
        const float learningRate = argumentCount > 4 ? std::stof(arguments[4]) : 0.05F;
        if (epochs == 0 || batchSize == 0 || learningRate <= 0) {
            throw std::runtime_error{"Epochs, batch size, and learning rate must be positive"};
        }

        std::cout << "Loading MNIST...\n";
        const MnistDataset trainingData =
            loadMnistDataset(datasetDirectory / "train-images-idx3-ubyte",
                             datasetDirectory / "train-labels-idx1-ubyte");
        const MnistDataset testData = loadMnistDataset(datasetDirectory / "t10k-images-idx3-ubyte",
                                                       datasetDirectory / "t10k-labels-idx1-ubyte");

        Sequential<float> model{
            Linear<float>{trainingData.pixelsPerImage(), hiddenSize},
            Activations::Tanh<float>{},
            Linear<float>{hiddenSize, numberOfClasses},
        };
        Optimizers::SGD<float> optimizer(model.parameters(), learningRate);
        Loss::CrossEntropy lossFunction;

        std::vector<size_t> trainingOrder(trainingData.size());
        std::iota(trainingOrder.begin(), trainingOrder.end(), size_t{0});
        std::mt19937 shuffleGenerator(42);

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            const auto epochStart = std::chrono::steady_clock::now();
            std::shuffle(trainingOrder.begin(), trainingOrder.end(), shuffleGenerator);
            float accumulatedLoss = 0;
            size_t correct = 0;

            for (size_t batchStart = 0; batchStart < trainingData.size(); batchStart += batchSize) {
                const size_t currentBatchSize =
                    std::min(batchSize, trainingData.size() - batchStart);
                Batch batch = makeBatch(trainingData, trainingOrder, batchStart, batchSize);

                optimizer.zeroGrad();
                Tensor<float> logits = model.forward(std::move(batch.images));
                correct += countCorrectPredictions(
                    logits, trainingData, trainingOrder, batchStart, currentBatchSize);
                Tensor<float> loss = lossFunction.forward(logits, batch.labels);
                accumulatedLoss += loss[0] * static_cast<float>(currentBatchSize);
                loss.backward();
                optimizer.step();
            }

            const float trainingLoss = accumulatedLoss / static_cast<float>(trainingData.size());
            const float trainingAccuracy =
                static_cast<float>(correct) / static_cast<float>(trainingData.size());
            const float testAccuracy = evaluate(model, testData, batchSize);
            const std::chrono::duration<double> elapsed =
                std::chrono::steady_clock::now() - epochStart;

            std::cout << std::fixed << std::setprecision(4) << "Epoch " << epoch + 1 << '/'
                      << epochs << "  loss=" << trainingLoss
                      << "  train_accuracy=" << trainingAccuracy * 100.0F << '%'
                      << "  test_accuracy=" << testAccuracy * 100.0F << '%'
                      << "  time=" << elapsed.count() << "s\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "MNIST example failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
