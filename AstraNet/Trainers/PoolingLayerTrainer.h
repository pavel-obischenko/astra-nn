//
// Created by Pavel Obischenko on 11/11/2016.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_POOLINGLAYERTRAINER_H
#define ASTRA_NN_POOLINGLAYERTRAINER_H

#include "LayerTrainer.h"

namespace astra {

    class PoolingLayerTrainer : public LayerTrainer {
    public:
        explicit PoolingLayerTrainer(const LayerPtr &layerPtr) : LayerTrainer(layerPtr) {}
        virtual const math::Vector& backpropagateError(const math::Vector& errorFactor, double epsilon, double momentum);
    };
}


#endif //ASTRA_NN_POOLINGLAYERTRAINER_H
