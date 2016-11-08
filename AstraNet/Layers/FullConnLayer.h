//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_FULLCONNLAYER_H
#define ASTRA_NN_FULLCONNLAYER_H

#include "Layer.h"

namespace astra {

    class FullConnLayer : public Layer {
    public:
        static LayerPtr createTanhLayerPtr(unsigned long nInputs, unsigned long nOutputs, double activationAlpha);
        static LayerPtr createSigmoidLayerPtr(unsigned long nInputs, unsigned long nOutputs, double activationAlpha);
        static LayerPtr createReLULayerPtr(unsigned long nInputs, unsigned long nOutputs);

    public:
        FullConnLayer(unsigned long nInputs, unsigned long nOutputs, const ActivationFunctionPtr& activationFunc);

    public:
        virtual LayerTrainerPtr createTrainer();
        virtual const math::Vector& process(const math::Vector& input);
    };

}


#endif //ASTRA_NN_FULLCONNLAYER_H
