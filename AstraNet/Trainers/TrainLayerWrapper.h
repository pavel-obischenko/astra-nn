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
#include "../Math/Vector.hpp"

namespace astra {
    
    struct TrainLayerWrapper {
        TrainLayerWrapper(const LayerPtr& layer, const VectorPtr& localGradient) : layer(layer), localGradient(localGradient) {}
        LayerPtr layer;
        VectorPtr localGradient;
    };
    
    typedef std::shared_ptr<TrainLayerWrapper> TrainLayerWrapperPtr;
    typedef std::vector<TrainLayerWrapperPtr> TrainLayerWrapperArray;
    typedef std::shared_ptr<TrainLayerWrapperArray> TrainLayerWrapperArrayPtr;
}

#endif /* TrainLayerWrapper_h */
