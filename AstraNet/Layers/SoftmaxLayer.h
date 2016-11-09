//
// Created by Pavel on 04/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_SOFTMAXLAYER_H
#define ASTRA_NN_SOFTMAXLAYER_H

#include "FullConnLayer.h"

namespace astra {

    class SoftmaxLayer : public FullConnLayer {
    public:
        static LayerPtr createSoftmaxLayerPtr(unsigned long nInputs, unsigned long nOutputs);

    public:
        SoftmaxLayer(unsigned long nInputs, unsigned long nOutputs);

    public:
        virtual LayerTrainerPtr createTrainer();
    };

}

#endif //ASTRA_NN_SOFTMAXLAYER_H
