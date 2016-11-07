//
// Created by Pavel on 07/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_SOFTMAXACTIVATIONFUNCTION_H
#define ASTRA_NN_SOFTMAXACTIVATIONFUNCTION_H

#include "ActivationFunction.h"
#include <cmath>

namespace astra {

    class SoftmaxActivationFunction : public ActivationFunction {
    public:
        virtual double value(double arg) const {
            return 1.;
        }

        virtual math::Vector value(const math::Vector& argV) const {
            math::Vector eV(argV.size());
            std::transform(argV.begin(), argV.end(), eV.begin(), [this](double arg) -> double {
                return std::exp(arg);
            });

            double sum = eV.sum();
            math::Vector result(argV.size());

            std::transform(eV.begin(), eV.end(), result.begin(), [this, sum](double arg) -> double {
                return arg / sum;
            });
            return result;
        }

        virtual double derivativeValue(double arg) const {
            return arg * (1 - arg);
        }

        virtual math::Vector derivativeValue(const math::Vector& argV) const {
            math::Vector output = value(argV);
            math::Vector result(argV.size());

            std::transform(output.begin(), output.end(), result.begin(), [this](double arg) -> double {
                return this->derivativeValue(arg);
            });
            return result;
        }
    };
}

#endif //ASTRA_NN_SOFTMAXACTIVATIONFUNCTION_H
