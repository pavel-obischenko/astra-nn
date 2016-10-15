//
//  TanhLayer.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "TanhLayer.h"
#include "../ActivationFunctions/TanhActivationFunction.h"

namespace astra {
    
    LayerPtr TanhLayer::createPtr(unsigned int nInputs, unsigned int nOutputs, double activationAlpha) {
        return std::make_shared<TanhLayer>(nInputs, nOutputs, activationAlpha);
    }
    
    TanhLayer::TanhLayer(unsigned int nInputs, unsigned int nOutputs, double activationAlpha) : astra::Layer(nInputs, nOutputs, std::make_shared<TanhActivationFunction>(activationAlpha)) {
    }
}
