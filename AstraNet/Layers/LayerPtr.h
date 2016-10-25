//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_LAYERPTR_H
#define ASTRA_NN_LAYERPTR_H

#include <memory>

namespace astra {

    class Layer;
    typedef std::shared_ptr<Layer> LayerPtr;

}


#endif //ASTRA_NN_LAYERPTR_H
