//
//  Layer.cpp
//  astra-nn
//
//  Created by Pavel on 20/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Layer.h"
#include <cassert>

using namespace astra::math;

namespace astra {

    Layer::Layer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr& activationFunc) : input(nInputs), output(nOutputs), weights(Matrix::rnd(nInputs + 1, nOutputs, -.5, .5)), activation(activationFunc) {}

    const Vector& Layer::process(const Vector& inputValues) {
        assert(activation != nullptr);
        assert(input.size() == inputValues.size());

        input = inputValues;
        output = activation->value(weights * InputVector(input).toVector());

        return output;
    }
}

