//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_LAYERPTR_H
#define ASTRA_NN_LAYERPTR_H

#include <memory>
#include "Layer.h"

namespace astra {

    class Layer;

    typedef std::shared_ptr<Layer> LayerPtr;
    typedef std::weak_ptr<Layer> LayerWeakPtr;

}


#endif //ASTRA_NN_LAYERPTR_H
