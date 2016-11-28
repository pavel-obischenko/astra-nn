//
// Created by Pavel Obischenko on 17/11/2016.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_POOLINGLAYERPTR_H
#define ASTRA_NN_POOLINGLAYERPTR_H

#include <memory>

namespace astra {

    class PoolingLayer;

    typedef std::shared_ptr<PoolingLayer> PoolingLayerPtr;
    typedef std::weak_ptr<PoolingLayer> PoolingLayerWeakPtr;
}

#endif //ASTRA_NN_POOLINGLAYERPTR_H
