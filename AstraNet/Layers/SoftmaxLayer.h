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
        static LayerPtr createSoftmaxLayerPtr(unsigned int nInputs, unsigned int nOutputs, double activationAlpha);

    public:
        SoftmaxLayer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr& activationFunc);

    public:
        virtual LayerTrainerPtr createTrainer();
        virtual const math::Vector& process(const math::Vector& input);
    };

}

#endif //ASTRA_NN_SOFTMAXLAYER_H
