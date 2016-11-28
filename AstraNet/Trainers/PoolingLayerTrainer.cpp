//
// Created by Pavel Obischenko on 11/11/2016.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "PoolingLayerTrainer.h"
#include "../Layers/PoolingLayer.h"
#include "../Algorithms/Image2Cols.h"

using namespace astra::math;
using namespace astra::algorithms;

namespace astra {

    const math::Vector& PoolingLayerTrainer::backpropagateError(const math::Vector& errorFactor, double epsilon, double momentum) {
        setPrevLayerErrorFactor(errorFactor);

        PoolingLayerPtr poolingLayerPtr = std::dynamic_pointer_cast<PoolingLayer>(getLayerPtr());

        std::vector<MatrixPtr> inputChannels = Image2Cols::matricesFromVector(poolingLayerPtr->getInput(), poolingLayerPtr->getNChannels());
        std::vector<MatrixPtr> errorChannels;

        auto src = errorFactor.begin();

        std::for_each(inputChannels.begin(), inputChannels.end(), [poolingLayerPtr, &src, &errorChannels](const MatrixPtr& channel) {
            MatrixPtr poolColsMatrix = Image2Cols::convertMatrixToColsForPooling(channel, poolingLayerPtr->getPoolingWidth(), poolingLayerPtr->getPoolingHeight());

            poolColsMatrix->for_each_col([&poolingLayerPtr, &src](MatrixPtr& col) {
                switch (poolingLayerPtr->getType()) {
                    case kPoolingLayerTypeMax: {
                        auto max = std::max_element(col->begin(), col->end());
                        col->zeroFill();
                        *max = *src;
                        break;
                    }
                    case kPoolingLayerTypeAverage: {
                        double err = *src / (double)col->size();
                        col->fill(err);
                        break;
                    }
                    default: break;
                }
                ++src;
            });

            errorChannels.push_back(poolColsMatrix);
        });

        // TODO: convert errorChannels to errorFactor

        return getErrorFactor();
    }
}
