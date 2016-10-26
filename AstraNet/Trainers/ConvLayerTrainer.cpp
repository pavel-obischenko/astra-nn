//
// Created by Pavel on 26/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "ConvLayerTrainer.h"

#include "../Math/Matrix.h"
#include "../Math/Vector.h"

using namespace astra::math;

namespace astra {

    const math::Vector& ConvLayerTrainer::backpropagateError(const math::Vector& prevLayerErrorFactor, double epsilon) {
        return getErrorFactor();
    }
}
