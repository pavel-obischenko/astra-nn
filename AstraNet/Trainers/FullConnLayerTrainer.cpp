//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "FullConnLayerTrainer.h"

using namespace astra::math;

namespace astra {

    const math::Vector& FullConnLayerTrainer::backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon) {
        setPrevLayerErrorFactor(prevLayerErrorFactor);

        const LayerPtr& layer = getLayerPtr();

        const ActivationFunctionPtr& activation = layer->getActivationFunc();
        const Matrix& weights = layer->getWeights();
        const InputVector& input = layer->getInput();

        // local gradient
        Vector derivatives = activation->derivativeValue(weights * input.toVector());
        Vector localGradient = derivatives.element_wise_mul(prevLayerErrorFactor);
        setLocalGradient(localGradient);

        // new weights
        setNewWeights(weights + input.toVector() * epsilon * localGradient);

        // error factor
        setErrorFactor(*(weights.transpose() * localGradient).head(weights.get_width() - 1));

        // save results
        layer->setWeights(getNewWeights());

        // return error factor
        return getErrorFactor();
    }
}
