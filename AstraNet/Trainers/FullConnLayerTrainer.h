//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_FULLCONNLAYERTRAINER_H
#define ASTRA_NN_FULLCONNLAYERTRAINER_H

#include "LayerTrainer.h"
#include "../Math/Math.h"

namespace astra {

    class FullConnLayerTrainer : public LayerTrainer {
    public:
        explicit FullConnLayerTrainer(const LayerPtr& layerPtr) : LayerTrainer(layerPtr) {}

    public:
        virtual const math::Vector& backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon, double momentum);
    };
}


#endif //ASTRA_NN_FULLCONNLAYERTRAINER_H
