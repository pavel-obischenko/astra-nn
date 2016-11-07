//
// Created by Pavel on 04/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_SOFTMAXLAYERTRAINER_H
#define ASTRA_NN_SOFTMAXLAYERTRAINER_H

#include "FullConnLayerTrainer.h"
#include "../Math/Math.h"

namespace astra {

    class SoftmaxLayerTrainer : public FullConnLayerTrainer  {
    public:
        explicit SoftmaxLayerTrainer(const LayerPtr& layerPtr) : FullConnLayerTrainer(layerPtr) {}

    public:
        virtual math::Vector trainingError(const math::Vector& out, const math::Vector& train);
    };

}

#endif //ASTRA_NN_SOFTMAXLAYERTRAINER_H
