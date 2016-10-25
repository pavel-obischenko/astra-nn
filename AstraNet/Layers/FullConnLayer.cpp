//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "FullConnLayer.h"

#include "../ActivationFunctions/TanhActivationFunction.h"
#include "../ActivationFunctions/SigActivationFunction.h"
#include "../ActivationFunctions/ReLUActivationFunction.h"

#include <cassert>

using namespace astra::math;

namespace astra {

    LayerPtr FullConnLayer::createTanhLayerPtr(unsigned int nInputs, unsigned int nOutputs, double activationAlpha) {
            return std::make_shared<FullConnLayer>(nInputs, nOutputs, std::make_shared<TanhActivationFunction>(activationAlpha));
    }

    LayerPtr FullConnLayer::createSigmoidLayerPtr(unsigned int nInputs, unsigned int nOutputs, double activationAlpha) {
            return std::make_shared<FullConnLayer>(nInputs, nOutputs, std::make_shared<SigActivationFunction>(activationAlpha));
    }

    LayerPtr FullConnLayer::createReLULayerPtr(unsigned int nInputs, unsigned int nOutputs) {
            return std::make_shared<FullConnLayer>(nInputs, nOutputs, std::make_shared<ReLUActivationFunction>());
    }

    FullConnLayer::FullConnLayer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr& activationFunc) : Layer(nInputs, nOutputs, activationFunc) {}

    LayerTrainerPtr FullConnLayer::createTrainer() {
        return nullptr;
    }

    const Vector& FullConnLayer::process(const Vector& inputValues) {
        assert(getActivationFunc() != nullptr);
        assert(getInput().size() == inputValues.size());

        setInput(inputValues);

        auto result = getWeights() * InputVector(getInput()).toVector();
        setOutput(getActivationFunc()->value(result));

        return getOutput();
    }

}