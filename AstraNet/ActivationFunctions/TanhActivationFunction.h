//
//  TanhActivationFunction.h
//  astra-nn
//
//  Created by Pavel on 27/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef TanhActivationFunction_h
#define TanhActivationFunction_h

#include "ActivationFunction.hpp"
#include <cmath>

namespace astra {
    
    class TanhActivationFunction : public ActivationFunction {
    public:
        TanhActivationFunction() : TanhActivationFunction(1) {}
        explicit TanhActivationFunction(double alpha) : alpha(alpha) {}
        
        virtual double value(double arg) const {
            return std::tanh(alpha * arg / 2.);
        }
        
        virtual double derivativeValue(double arg) const {
            double th = value(arg);
            return alpha * (1 - th * th) / 2.;
        }
        
    private:
        double alpha;
    };
    
}

#endif /* TanhActivationFunction_h */
