//
//  Layer.cpp
//  astra-nn
//
//  Created by Pavel on 20/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include <stdio.h>
#include <random>

#include "Layer.hpp"
#include "../Trainers/Trainer.hpp"

namespace astra {

    Layer::Layer(unsigned int nInputs, unsigned int nOutputs, const std::shared_ptr<ActivationFunction>& activationFunc) : input(nInputs + 1), output(nOutputs), weights(nOutputs, nInputs + 1), activation(activationFunc) {
        input[0] = 1.;
        initWeights();
    }

    const vec& Layer::process(const vec& inputValues) {
        if (!activation) {
            // TODO: throw exception
        }
        
        if (Layer::input.size() != inputValues.size() + 1) {
            // TODO: throw exception
        }

        input.fill_from(inputValues);
        
        output = weights * input;
        
        std::vector<double> result;
        std::for_each(output.get_storage_const().begin(), output.get_storage_const().end(), [this, &result](const double& val) {
            result.push_back(this->activation->value(val));
        });
        
        output = vec(result);
                
        return output;
    }
    
    void Layer::train(Trainer* trainer, double epsilon) {
        trainer->trainLayer(this, epsilon);
    }
    
    void Layer::initWeights() {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-.5, .5);
        auto rnd = std::bind(distribution, generator);
        
        std::for_each(weights.get_rows().begin(), weights.get_rows().end(), [&rnd](vec& row) {
            std::for_each(row.get_storage().begin(), row.get_storage().end(), [&rnd](double& val) {
                val = rnd();
            });
        });
    }

}
