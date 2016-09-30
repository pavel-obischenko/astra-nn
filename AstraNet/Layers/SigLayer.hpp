//
//  SigLayer.hpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef SigLayer_hpp
#define SigLayer_hpp

#include "Layer.hpp"

namespace astra {
    
    class SigLayer : public Layer {
    public:
        SigLayer(unsigned int nInputs, unsigned int nOutputs, double activationAlpha = 1.);
    };
}

#endif /* SigLayer_hpp */
