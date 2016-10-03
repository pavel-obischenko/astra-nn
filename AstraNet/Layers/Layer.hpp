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
    
    class Trainer;
    
    class Layer {
    private:
        Layer() : weights(1, 1) {};
        
    protected:
        Layer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr &activationFunc);
        
    public:
        const vec& process(const vec& inputValues);
        
    public:
        void setWeights(const mat& newWeights) { weights = newWeights; }
        const inputv& getInput() const { return input; }
        const mat& getWeights() const { return weights; }
        const vec& getOutput() const { return output; }
        const ActivationFunctionPtr& getActivationFunc() const { return activation; }
        
    protected:
        void initWeights();
        
    protected:
        inputv input;
        vec output;
        mat weights;
        
        ActivationFunctionPtr activation;
    };
    
    typedef std::shared_ptr<Layer> LayerPtr;
}

#endif /* Layer_hpp */
