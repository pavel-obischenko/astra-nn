//
// Created by Pavel on 04/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "SoftmaxLayer.h"
#include "../ActivationFunctions/SigActivationFunction.h"
#include "../Trainers/SoftmaxLayerTrainer.h"
#include "../Math/Math.h"

using namespace astra::math;

namespace astra {

    LayerPtr SoftmaxLayer::createSoftmaxLayerPtr(unsigned int nInputs, unsigned int nOutputs, double activationAlpha) {
        return std::make_shared<SoftmaxLayer>(nInputs, nOutputs, std::make_shared<SigActivationFunction>(activationAlpha));
    }

    SoftmaxLayer::SoftmaxLayer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr& activationFunc) : FullConnLayer(nInputs, nOutputs, activationFunc) {}

    LayerTrainerPtr SoftmaxLayer::createTrainer() {
        return std::make_shared<SoftmaxLayerTrainer>(shared_from_this());
    }

    const math::Vector& SoftmaxLayer::process(const math::Vector& input) {
        Vector result = FullConnLayer::process(input);
        // TODO: add softmax
        return getOutput();
    }
}
