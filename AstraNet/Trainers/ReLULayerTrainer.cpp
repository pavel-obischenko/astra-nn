//
// Created by Pavel on 09/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "ReLULayerTrainer.h"
#include "../ActivationFunctions/ActivationFunction.h"

using namespace astra::math;

namespace astra {

    const math::Vector& ReLULayerTrainer::backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon, double momentum) {
        setPrevLayerErrorFactor(prevLayerErrorFactor);
        LayerPtr layer = getLayerPtr();
        const Vector& input = InputVector(layer->getInput()).toVector();
        ActivationFunctionPtr activation = layer->getActivationFunc();

        auto derivatives = activation->derivativeValue(*input.head(input.size() - 1));
        setErrorFactor(derivatives.element_wise_mul(prevLayerErrorFactor));

        return getErrorFactor();
    }
}
