//
// Created by Pavel on 09/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_RELULAYERTRAINER_H
#define ASTRA_NN_RELULAYERTRAINER_H

#include "LayerTrainer.h"

namespace astra {

    class ReLULayerTrainer : public LayerTrainer {
    public:
        explicit ReLULayerTrainer(const LayerPtr& layerPtr) : LayerTrainer(layerPtr) {}

    public:
        virtual const math::Vector& backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon, double momentum);
    };

}


#endif //ASTRA_NN_RELULAYERTRAINER_H
