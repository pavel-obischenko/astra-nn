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

namespace astra {
    
    class ActivationFunction {
    public:
        virtual double value(double arg) const { return arg; }
        virtual double derivativeValue(double arg) { return 0; }
    };

}

#endif /* IActivationFunction_h */
