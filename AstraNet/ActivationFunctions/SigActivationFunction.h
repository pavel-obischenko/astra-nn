//
//  SigActivationFunction.h
//  astra-nn
//
//  Created by Pavel on 28/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef SigActivationFunction_h
#define SigActivationFunction_h

#include "ActivationFunction.hpp"
#include <cmath>

namespace astra {
    
    class SigActivationFunction : public ActivationFunction {
    public:
        SigActivationFunction() : SigActivationFunction(1) {}
        explicit SigActivationFunction(double alpha) : alpha(alpha) {}
        
        virtual double value(double arg) const {
            return 1 / (1 + std::exp(-alpha * arg));
        }
        
        virtual double derivativeValue(double arg) const {
            double s = value(arg);
            return alpha * s * (1 - s);
        }
        
    private:
        double alpha;
    };
    
}


#endif /* SigActivationFunction_h */
