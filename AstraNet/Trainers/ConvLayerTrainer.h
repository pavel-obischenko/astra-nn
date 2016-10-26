//
// Created by Pavel on 26/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_CONVLAYERTRAINER_H
#define ASTRA_NN_CONVLAYERTRAINER_H

#include "LayerTrainer.h"
#include "../Math/Math.h"

namespace astra {

    class ConvLayerTrainer : public LayerTrainer {
    public:
        explicit ConvLayerTrainer(const LayerPtr& layerPtr) : LayerTrainer(layerPtr) {}

    public:
        virtual const math::Vector& backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon);
    };

}


#endif //ASTRA_NN_CONVLAYERTRAINER_H
