//
// Created by Pavel on 17/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_RELUACTIVATIONFUNCTION_H
#define ASTRA_NN_RELUACTIVATIONFUNCTION_H

#include "ActivationFunction.h"
#include <algorithm>

namespace astra {

    class ReLUActivationFunction : public ActivationFunction {
    public:
        ReLUActivationFunction() {}
        virtual double value(double arg) const {
            return std::max(arg, 0.0);
        }
        virtual double derivativeValue(double arg) const {
            return arg < 0 ? 0 : 1.;
        }
    };

}


#endif //ASTRA_NN_RELUACTIVATIONFUNCTION_H
