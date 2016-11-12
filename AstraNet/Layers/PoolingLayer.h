//
// Created by Pavel Obischenko on 11/11/2016.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_POOLINGLAYER_H
#define ASTRA_NN_POOLINGLAYER_H

#include "Layer.h"

namespace astra {

    class PoolingLayer : public Layer {
    public:
        PoolingLayer(unsigned long width, unsigned long height, unsigned long poolingWidth, unsigned long poolingHeight) {}

    public:
        virtual LayerTrainerPtr createTrainer();
        virtual const math::Vector& process(const math::Vector& input);
    };
}


#endif //ASTRA_NN_POOLINGLAYER_H
