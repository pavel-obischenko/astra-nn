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
#include "../Math/Math.h"

#include <memory>

namespace astra {

    class Layer : public std::enable_shared_from_this<Layer> {
    protected:
        Layer() : nInputs(2), nOutputs(1), input(2), output(1), weights(2, 2), activation(nullptr) {};
        Layer(unsigned long nInputs, unsigned long nOutputs, const ActivationFunctionPtr& activationFunc) : nInputs(nInputs), nOutputs(nOutputs), input(nInputs), output(nOutputs), weights(math::Matrix::rnd(nInputs + 1, nOutputs, -.15, .15)), activation(activationFunc) {}
        virtual ~Layer() {}
        
    public:
        virtual LayerTrainerPtr createTrainer() = 0;
        virtual const math::Vector& process(const math::Vector& input) = 0;
        
    public:
        unsigned long getNInputs() const { return nInputs; }
        unsigned long getNOutputs() const { return nOutputs; }

        void setWeights(const math::Matrix& w) { weights = w; }

        const math::Vector& getInput() const { return input; }
        const math::Matrix& getWeights() const { return weights; }
        const math::Vector& getOutput() const { return output; }

        const ActivationFunctionPtr& getActivationFunc() const { return activation; }

    protected:
        void setInput(const math::Vector& i) { input = i; }
        void setOutput(const math::Vector& o) { output = o; }
        void setActivationFunc(const ActivationFunctionPtr& af) { activation = af; }
        
    private:
        unsigned long nInputs;
        unsigned long nOutputs;

        math::Vector input;
        math::Vector output;
        math::Matrix weights;
        
        ActivationFunctionPtr activation;
    };
}

#endif /* ASTRA_NN_LAYER_H */
