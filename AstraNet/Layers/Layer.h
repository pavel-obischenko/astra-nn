//
//  Layer.hpp
//  astra-nn
//
//  Created by Pavel on 20/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef ASTRA_NN_LAYER_H
#define ASTRA_NN_LAYER_H

#include "LayerPtr.h"
#include "../Trainers/LayerTrainerPtr.h"

#include "../ActivationFunctions/ActivationFunction.h"

#include "../Math/Vector.h"
#include "../Math/Matrix.h"
#include "../Math/InputVector.h"

namespace astra {

    class Layer {
    protected:
        Layer() : input(2), output(1), weights(2, 2), activation(nullptr) {};
        Layer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr& activationFunc) : input(nInputs), output(nOutputs), weights(math::Matrix::rnd(nInputs + 1, nOutputs, -.5, .5)), activation(activationFunc) {}
        virtual ~Layer() {}
        
    public:
        virtual LayerTrainerPtr createTrainer() = 0;
        virtual const math::Vector& process(const math::Vector& input) = 0;
        
    public:
        void setWeights(const math::Matrix& w) { weights = w; }

        const math::Vector& getInput() const { return input; }
        const math::Matrix& getWeights() const { return weights; }
        const math::Vector& getOutput() const { return output; }

        const ActivationFunctionPtr& getActivationFunc() const { return activation; }

    protected:
        void setInput(const math::Vector& i) { input = i; }
        void setOutput(const math::Vector& o) { output = o; }
        
    protected:
        math::Vector input;
        math::Vector output;
        math::Matrix weights;
        
        ActivationFunctionPtr activation;
    };
}

#endif /* ASTRA_NN_LAYER_H */
