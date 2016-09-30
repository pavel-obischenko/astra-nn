//
//  SigLayer.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "SigLayer.hpp"
#include "../ActivationFunctions/SigActivationFunction.h"

namespace astra {
    
    SigLayer::SigLayer(unsigned int nInputs, unsigned int nOutputs, double activationAlpha) : astra::Layer(nInputs, nOutputs, std::make_shared<SigActivationFunction>(activationAlpha)) {
    }
}
