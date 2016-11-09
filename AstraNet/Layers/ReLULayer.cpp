//
// Created by Pavel on 09/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "ReLULayer.h"
#include "../ActivationFunctions/ReLUActivationFunction.h"
#include "../Trainers/ReLULayerTrainer.h"
#include <memory>

namespace astra {

    LayerPtr ReLULayer::createReLULayerPtr(unsigned long size) {
        return std::make_shared<ReLULayer>(size);
    }

    ReLULayer::ReLULayer(unsigned long size) : Layer(size, size, std::make_shared<ReLUActivationFunction>()) {}

    const math::Vector& ReLULayer::process(const math::Vector& input) {
        setInput(input);

        ActivationFunctionPtr activation = getActivationFunc();
        setOutput(activation->value(input));

        return getOutput();
    }

    LayerTrainerPtr ReLULayer::createTrainer() {
        return std::make_shared<ReLULayerTrainer>(shared_from_this());
    }
}
