//
//  IActivationFunction.hpp
//  astra-nn
//
//  Created by Pavel on 20/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef ActivationFunction_hpp
#define ActivationFunction_hpp

#include "../Math/Vector.h"

#include <vector>
#include <algorithm>

namespace astra {
    
    class ActivationFunction {
    public:
        virtual double value(double arg) const { return arg; }
        virtual double derivativeValue(double arg) const { return 0; }

        virtual math::Matrix value(const math::Matrix& argM) const {
            math::Matrix result(argM.get_width(), argM.get_height());
            std::transform(argM.begin(), argM.end(), result.begin(), [this](double arg) -> double {
                return this->value(arg);
            });
            return result;
        }

        virtual math::Vector value(const math::Vector& argV) const {
            math::Vector result(argV.size());
            std::transform(argV.begin(), argV.end(), result.begin(), [this](double arg) -> double {
                return this->value(arg);
            });
            return result;
        }

        virtual math::Matrix derivativeValue(const math::Matrix& argM) const {
            math::Matrix result(argM.get_width(), argM.get_height());
            std::transform(argM.begin(), argM.end(), result.begin(), [this](double arg) -> double {
                return this->derivativeValue(arg);
            });
            return result;
        }
        
        virtual math::Vector derivativeValue(const math::Vector& argV) const {
            math::Vector result(argV.size());
            std::transform(argV.begin(), argV.end(), result.begin(), [this](double arg) -> double {
                return this->derivativeValue(arg);
            });
            return result;
        }
    };

    typedef std::shared_ptr<ActivationFunction> ActivationFunctionPtr;
}

#endif /* IActivationFunction_h */
