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

    Layer::Layer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr& activationFunc) : input(nInputs + 1), output(nOutputs), weights(Matrix::rnd(nInputs + 1, nOutputs, -.5, .5)), activation(activationFunc) {}

    const Vector& Layer::process(const InputVector& inputValues) {
        assert(activation != nullptr);
        assert(Layer::input.size() == inputValues.size());

        input = inputValues;
        output = activation->value(weights * input.toVector()); // TODO: REFACTOR IT !!!

        return output;
    }
}

