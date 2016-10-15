//
//  Layer.cpp
//  astra-nn
//
//  Created by Pavel on 20/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include <random>
#include "Layer.h"

using namespace astra::math;

namespace astra {

    Layer::Layer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr& activationFunc) : input(nInputs + 1), output(nOutputs), weights(nInputs + 1, nOutputs), activation(activationFunc) {
        initWeights();
    }

    const Vector& Layer::process(const InputVector& inputValues) {
        assert(activation != nullptr);
        assert(Layer::input.size() == inputValues.size());

        input = inputValues;
        output = activation->value(weights * input.toVector()); // TODO: REFACTOR IT !!!

        return output;
    }
    
    void Layer::initWeights() {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-.5, .5);
        auto rnd = std::bind(distribution, generator);

        weights.for_each([&rnd](double &val) {
            val = rnd();
        });
    }
}

