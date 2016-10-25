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
        static LayerPtr createTanhLayerPtr(unsigned int nInputs, unsigned int nOutputs, double activationAlpha);
        static LayerPtr createSigmoidLayerPtr(unsigned int nInputs, unsigned int nOutputs, double activationAlpha);
        static LayerPtr createReLULayerPtr(unsigned int nInputs, unsigned int nOutputs);

    public:
        FullConnLayer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr& activationFunc);

    public:
        virtual const math::Vector& process(const math::Vector& input);
    };

}


#endif //ASTRA_NN_FULLCONNLAYER_H
