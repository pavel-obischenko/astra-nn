//
//  TanhLayer.hpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef TanhLayer_hpp
#define TanhLayer_hpp

#include "Layer.hpp"

namespace astra {
    
    class TanhLayer : public Layer {
    public:
        TanhLayer(unsigned int nInputs, unsigned int nOutputs, double activationAlpha = 1.);
    };
}

#endif /* TanhLayer_hpp */
