//
// Created by Pavel on 04/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "SoftmaxLayer.h"
#include "../ActivationFunctions/SoftmaxActivationFunction.h"
#include "../Trainers/SoftmaxLayerTrainer.h"
#include "../Math/Math.h"

using namespace astra::math;

namespace astra {

    LayerPtr SoftmaxLayer::createSoftmaxLayerPtr(unsigned int nInputs, unsigned int nOutputs) {
        return std::make_shared<SoftmaxLayer>(nInputs, nOutputs);
    }

    SoftmaxLayer::SoftmaxLayer(unsigned int nInputs, unsigned int nOutputs) : FullConnLayer(nInputs, nOutputs, std::make_shared<SoftmaxActivationFunction>()) {}

//    LayerTrainerPtr SoftmaxLayer::createTrainer() {
//        return std::make_shared<SoftmaxLayerTrainer>(shared_from_this());
//    }
}
