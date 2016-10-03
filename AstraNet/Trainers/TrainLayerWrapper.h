//
//  TrainLayerWrapper.h
//  astra-nn
//
//  Created by Pavel on 03/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef TrainLayerWrapper_h
#define TrainLayerWrapper_h

#include "../Layers/Layer.hpp"

namespace astra {
    
    struct TrainLayerWrapper {
        LayerPtr layer;
        Matrix weightGradient;
    };
}

#endif /* TrainLayerWrapper_h */
