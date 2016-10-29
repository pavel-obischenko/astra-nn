//
// Created by Pavel on 26/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "ConvLayerTrainer.h"

#include "../Math/Math.h"

#include "../Algorithms/Image2Cols.h"
#include "../Layers/ConvLayer.h"

using namespace astra::math;
using namespace astra::algorithms;

namespace astra {

    const math::Vector& ConvLayerTrainer::backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon) {
        setPrevLayerErrorFactor(prevLayerErrorFactor);

        ConvLayerPtr convLayerPtr = std::dynamic_pointer_cast<ConvLayer>(getLayerPtr());

        const ActivationFunctionPtr& activation = convLayerPtr->getActivationFunc();
        auto inputCols = convLayerPtr->getInputCols();
        auto filters = convLayerPtr->getWeights();
        auto linearResult = convLayerPtr->getLinearMultiplicationResult();

        auto errorMat = prevLayerErrorFactor.toMatrix(filters.get_width() - 1, filters.get_height());

        // local gradient
        auto derivatives = activation->derivativeValue(linearResult);
        auto localGradient = derivatives.element_wise_mul(errorMat);
        setLocalGradient(localGradient.toVector());

        // new weights
        setNewWeights(filters + (localGradient * inputCols.transpose()) * epsilon);

        // error factor
        auto backErrFactorCols =  filters.transpose() * localGradient;
        // TODO: convert error factor cols to input vector

        // save results
        convLayerPtr->setWeights(getNewWeights());

        // return error factor
        return getErrorFactor();
    }
}
