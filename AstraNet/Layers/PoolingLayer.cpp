//
// Created by Pavel Obischenko on 11/11/2016.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "PoolingLayer.h"
#include "../Trainers/PoolingLayerTrainer.h"
#include <memory>

namespace astra {

    LayerTrainerPtr PoolingLayer::createTrainer() {
        return std::make_shared<PoolingLayerTrainer>(shared_from_this());
    }

    const math::Vector& PoolingLayer::process(const math::Vector& input) {
        return getOutput();
    }

}
