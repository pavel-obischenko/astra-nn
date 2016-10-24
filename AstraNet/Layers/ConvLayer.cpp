//
// Created by Pavel on 23/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "ConvLayer.h"
#include "../Algorithms/Image2Cols.h"

using namespace astra::math;
using namespace astra::algorithms;

namespace astra {

    ConvLayer::ConvLayer(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long filterWidth, unsigned long filterHeight, unsigned long nFilters) {
        unsigned long kernelsCountH = Image2Cols::kernelsCount(width, filterWidth);
        unsigned long kernelsCountV = Image2Cols::kernelsCount(height, filterHeight);


    }

    const Vector& ConvLayer::process(const Vector& input) {
        return Layer::getOutput();
    }

}