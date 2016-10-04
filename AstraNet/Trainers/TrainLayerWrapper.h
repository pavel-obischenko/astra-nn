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
        TrainLayerWrapper(LayerPtr& layer, MatrixPtr& weightGradient) : layer(layer), weightGradient(weightGradient) {}
        LayerPtr layer;
        MatrixPtr weightGradient;
    };
    
    typedef std::shared_ptr<TrainLayerWrapper> TrainLayerWrapperPtr;
    typedef std::vector<TrainLayerWrapperPtr> TrainLayerWrapperArray;
    typedef std::shared_ptr<TrainLayerWrapperArray> TrainLayerWrapperArrayPtr;
}

#endif /* TrainLayerWrapper_h */
