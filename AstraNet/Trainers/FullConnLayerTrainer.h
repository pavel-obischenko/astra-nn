//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_FULLCONNLAYERTRAINER_H
#define ASTRA_NN_FULLCONNLAYERTRAINER_H

#include "LayerTrainer.h"

namespace astra {

    class FullConnLayerTrainer : public LayerTrainer {
    public:
        explicit FullConnLayerTrainer(const LayerPtr& layerPtr) : LayerTrainer(layerPtr) {}

    public:
        virtual math::Vector trainingError(const math::Vector& out, const math::Vector& train);
        virtual const math::Vector& backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon);
    };

}


#endif //ASTRA_NN_FULLCONNLAYERTRAINER_H
