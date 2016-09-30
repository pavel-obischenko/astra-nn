//
//  Layer.hpp
//  astra-nn
//
//  Created by Pavel on 20/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include "../ActivationFunctions/ActivationFunction.hpp"

#include "../Math/Vector.hpp"
#include "../Math/Matrix.hpp"
#include "../Math/InputVector.hpp"

namespace astra {
    
    class Layer {
    private:
        Layer() : weights(1, 1) {};
        
    public:
        Layer(unsigned int nInputs, unsigned int nOutputs, const std::shared_ptr<ActivationFunction> &activationFunc);
        const vec& process(const vec& inputValues);
        
    public:
        void setWeights(const mat& newWeights) { weights = newWeights; }
        const mat& getWeights() const { return weights; }
        
        const vec& getOutput() const { return output; }
        
    protected:
        void initWeights();
        
    protected:
        inputv input;
        vec output;
        mat weights;
        
        std::shared_ptr<ActivationFunction> activation;
    };
}

#endif /* Layer_hpp */
