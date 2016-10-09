//
//  IActivationFunction.hpp
//  astra-nn
//
//  Created by Pavel on 20/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef ActivationFunction_hpp
#define ActivationFunction_hpp

#include "../Math/Vector.hpp"

#include <vector>
#include <algorithm>

namespace astra {
    
    class ActivationFunction {
    public:
        virtual double value(double arg) const { return arg; }
        virtual double derivativeValue(double arg) const { return 0; }
        
        virtual Vector value(const Vector& argV) const {
            std::vector<double> result;
            std::for_each(argV.get_storage().begin(), argV.get_storage().end(), [this, &result](double arg) {
                result.push_back(this->value(arg));
            });
            return Vector(result);
        }
        
        virtual Vector derivativeValue(const Vector& argV) const {
            std::vector<double> result;
            std::for_each(argV.get_storage().begin(), argV.get_storage().end(), [this, &result](double arg) {
                result.push_back(this->derivativeValue(arg));
            });
            return Vector(result);
        }
    };

    typedef std::shared_ptr<ActivationFunction> ActivationFunctionPtr;
}

#endif /* IActivationFunction_h */
