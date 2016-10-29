//
// Created by Pavel on 26/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_CONVLAYERPTR_H
#define ASTRA_NN_CONVLAYERPTR_H

#include <memory>

namespace astra {

    class ConvLayer;

    typedef std::shared_ptr<ConvLayer> ConvLayerPtr;
    typedef std::weak_ptr<ConvLayer> ConvLayerWeakPtr;
}

#endif //ASTRA_NN_CONVLAYERPTR_H