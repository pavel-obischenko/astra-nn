//
// Created by Pavel Obischenko on 11/11/2016.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "PoolingLayerTrainer.h"

namespace astra {

    const math::Vector& PoolingLayerTrainer::backpropagateError(const math::Vector& errorFactor, double epsilon, double momentum) {
        return errorFactor;
    }
}
