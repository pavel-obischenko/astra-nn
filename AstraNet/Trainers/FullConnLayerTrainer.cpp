//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "FullConnLayerTrainer.h"

using namespace astra::math;

namespace astra {

    const math::Vector& FullConnLayerTrainer::backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon, double momentum) {
        setPrevLayerErrorFactor(prevLayerErrorFactor);

        const LayerPtr& layer = getLayerPtr();

        const ActivationFunctionPtr& activation = layer->getActivationFunc();
        const Matrix& weights = layer->getWeights();
        const Vector& input = InputVector(layer->getInput()).toVector();

        // local gradient
        Vector derivatives = activation->derivativeValue(weights * input);
        Vector localGradient = derivatives.element_wise_mul(prevLayerErrorFactor);
        setLocalGradient(localGradient);

        // new weights
        auto dWeights = epsilon * localGradient.toMatrix() * input.transpose();
        adjustWeights(dWeights, momentum);
        setDWeights(dWeights);

        // error factor
        setErrorFactor(*(weights.transpose() * localGradient).head(weights.get_width() - 1));

        // save results
        layer->setWeights(getNewWeights());

        // return error factor
        return getErrorFactor();
    }
}
