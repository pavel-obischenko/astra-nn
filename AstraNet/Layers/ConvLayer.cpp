//
// Created by Pavel on 23/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "ConvLayer.h"
#include "../Algorithms/Image2Cols.h"
#include "../Trainers/ConvLayerTrainer.h"

using namespace astra::math;
using namespace astra::algorithms;

namespace astra {

    LayerPtr ConvLayer::createConvLayerPtr(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long filterWidth, unsigned long filterHeight, unsigned long nFilters) {
        return std::make_shared<ConvLayer>(width, height, nChannels, filterWidth, filterHeight, nFilters);
    }

    ConvLayer::ConvLayer(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long filterWidth, unsigned long filterHeight, unsigned long nFilters) : width(width), height(height), nChannels(nChannels), filterWidth(filterWidth), filterHeight(filterHeight), nFilters(nFilters) {
        unsigned long kernelsCountH = Image2Cols::kernelsCount(width, filterWidth);
        unsigned long kernelsCountV = Image2Cols::kernelsCount(height, filterHeight);

        setInput(Vector(width * height * nChannels));
        setOutput(Vector(kernelsCountH * kernelsCountV * nFilters));

        // width = channels * filter_size + bias
        // height = filters qty
        setWeights(Matrix::rnd(nChannels * filterWidth * filterHeight + 1, nFilters));
    }

    LayerTrainerPtr ConvLayer::createTrainer() {
        return std::make_shared<ConvLayerTrainer>(shared_from_this());
    }

    const Vector& ConvLayer::process(const Vector& input) {
        setInput(input);
        Matrix inputCols = *Image2Cols::convertToCols(input, nChannels, filterWidth, filterHeight, true);
        setInputCols(inputCols);

        auto result = getWeights() * inputCols;
        setLinearMultiplicationResult(result);

        ActivationFunctionPtr activation = getActivationFunc();
        auto output = activation != nullptr ? activation->value(result).toVector() : result.toVector();
        setOutput(output);

        return Layer::getOutput();
    }
}
