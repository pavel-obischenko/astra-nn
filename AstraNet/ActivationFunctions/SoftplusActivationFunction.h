//
// Created by Pavel on 17/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_SOFTPLUSATIONFUNCTION_H
#define ASTRA_NN_SOFTPLUSATIONFUNCTION_H

#include "ActivationFunction.h"
#include <cmath>

namespace astra {

    class SoftplusActivationFunction : public ActivationFunction {
    public:
        SoftplusActivationFunction() {}
        virtual double value(double arg) const {
            return  std::log(1 + std::exp(arg));
        }

        virtual double derivativeValue(double arg) const {
            return 1 / (1 + std::exp(-arg));
        }
    };
}

#endif //ASTRA_NN_RELUACTIVATIONFUNCTION_H
