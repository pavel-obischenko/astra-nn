//
// Created by Pavel on 09/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_RELULAYER_H
#define ASTRA_NN_RELULAYER_H

#include "Layer.h"

namespace astra {

    class ReLULayer : public Layer {
    public:
        static LayerPtr createReLULayerPtr(unsigned long size);

    public:
        ReLULayer(unsigned long size);

    public:
        virtual const math::Vector& process(const math::Vector& input);
        virtual LayerTrainerPtr createTrainer();
    };

}


#endif //ASTRA_NN_RELULAYER_H
