//
// Created by Pavel Obischenko on 11/11/2016.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "PoolingLayer.h"
#include "../Trainers/PoolingLayerTrainer.h"
#include "../Algorithms/Image2Cols.h"
#include "../Math/Math.h"

#include <memory>

using namespace astra::math;
using namespace astra::algorithms;

namespace astra {

    LayerPtr PoolingLayer::createMaxPoolingLayerPtr(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long poolWidth, unsigned long poolHeight) {
        return PoolingLayer::createPoolingLayerPtr(kPoolingLayerTypeMax, width, height, nChannels, poolWidth, poolHeight);;
    }

    LayerPtr PoolingLayer::createAveragePoolingLayerPtr(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long poolWidth, unsigned long poolHeight) {
        return PoolingLayer::createPoolingLayerPtr(kPoolingLayerTypeAverage, width, height, nChannels, poolWidth, poolHeight);
    }

    LayerPtr PoolingLayer::createPoolingLayerPtr(PoolingLayerType type, unsigned long width, unsigned long height, unsigned long nChannels, unsigned long poolWidth, unsigned long poolHeight) {
        return std::make_shared<PoolingLayer>(type, width, height, nChannels, poolWidth, poolHeight);
    }

    PoolingLayer::PoolingLayer(PoolingLayerType type, unsigned long width, unsigned long height, unsigned long nChannels, unsigned long poolingWidth, unsigned long poolingHeight) : type(type), width(width), height(height), nChannels(nChannels), poolingWidth(poolingWidth), poolingHeight(poolingHeight) {
        unsigned long poolsCountH = Image2Cols::poolsCount(width, poolingWidth);
        unsigned long poolsCountV = Image2Cols::poolsCount(height, poolingHeight);;

        setInput(Vector(width * height * nChannels));
        setOutput(Vector(poolsCountH * poolsCountV * nChannels));
    }

    LayerTrainerPtr PoolingLayer::createTrainer() {
        return std::make_shared<PoolingLayerTrainer>(shared_from_this());
    }

    const math::Vector& PoolingLayer::process(const math::Vector& input) {
        std::vector<MatrixPtr> inputChannels = Image2Cols::matricesFromVector(input, getNChannels());

        Vector output(getOutput().size());
        auto dst = output.begin();

        std::for_each(inputChannels.begin(), inputChannels.end(), [this, &dst](const MatrixPtr& channel) {
            MatrixPtr poolColsMatrix = Image2Cols::convertMatrixToColsForPooling(channel, this->getPoolingWidth(), this->getPoolingHeight());

            poolColsMatrix->for_each_col([this, &dst](const MatrixPtr& col) {
                switch (this->getType()) {
                    case kPoolingLayerTypeMax:
                        *dst = col->max_element();
                        break;
                    case kPoolingLayerTypeAverage:
                        *dst = col->average_value();
                        break;
                    default: break;
                }
                ++dst;
            });
        });

        setOutput(output);
        return getOutput();
    }
}
