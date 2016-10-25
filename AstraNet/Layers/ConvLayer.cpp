//
// Created by Pavel on 23/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "ConvLayer.h"
#include "../Algorithms/Image2Cols.h"

using namespace astra::math;
using namespace astra::algorithms;

namespace astra {

    ConvLayer::ConvLayer(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long filterWidth, unsigned long filterHeight, unsigned long nFilters) : width(width), height(height), nChannels(nChannels), filterWidth(filterWidth), filterHeight(filterHeight), nFilters(nFilters) {
        unsigned long kernelsCountH = Image2Cols::kernelsCount(width, filterWidth);
        unsigned long kernelsCountV = Image2Cols::kernelsCount(height, filterHeight);

        setInput(Vector(kernelsCountH * kernelsCountV * nChannels));

        // width = filter size + bias
        // height = filters qty
        setWeights(Matrix::rnd(filterWidth * filterHeight + 1, nFilters));
    }

    const Vector& ConvLayer::process(const Vector& input) {
        setInput(input);
        MatrixPtr inputCols = Image2Cols::convertToCols(input, nChannels, filterWidth, filterHeight, true);

        auto result = getWeights() * (*inputCols);
        setOutput(getActivationFunc()->value(result).toVector());

        return Layer::getOutput();
    }
}