//
//  TrainLayerWrapper.h
//  astra-nn
//
//  Created by Pavel on 03/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef TrainLayerWrapper_h
#define TrainLayerWrapper_h

#include "../Layers/Layer.h"

#include "../Math/Vector.h"
#include "../Math/Matrix.h"

namespace astra {
    
    struct TrainLayerWrapper {
        TrainLayerWrapper(const LayerPtr& layer, const astra::math::VectorPtr& localGradient) : layer(layer), localGradient(localGradient) {}
        LayerPtr layer;
        astra::math::VectorPtr localGradient;
        astra::math::MatrixPtr newWeights;
    };
    
    typedef std::shared_ptr<TrainLayerWrapper> TrainLayerWrapperPtr;
    typedef std::vector<TrainLayerWrapperPtr> TrainLayerWrapperArray;
    typedef std::shared_ptr<TrainLayerWrapperArray> TrainLayerWrapperArrayPtr;
}

#endif /* TrainLayerWrapper_h */
