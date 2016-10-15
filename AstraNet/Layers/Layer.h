//
//  Layer.hpp
//  astra-nn
//
//  Created by Pavel on 20/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include "../ActivationFunctions/ActivationFunction.h"

#include "../Math/Vector.h"
#include "../Math/Matrix.h"
#include "../Math/InputVector.h"

namespace astra {

    class Layer {
    private:
        Layer() : input(2), output(1), weights(2, 2), activation(nullptr) {};
        
    protected:
        Layer(unsigned int nInputs, unsigned int nOutputs, const ActivationFunctionPtr &activationFunc);
        
    public:
        const math::Vector& process(const math::InputVector& input);
        
    public:
        void setWeights(const math::Matrix& newWeights) { weights = newWeights; }
        const math::InputVector& getInput() const { return input; }
        const math::Matrix& getWeights() const { return weights; }
        const math::Vector& getOutput() const { return output; }
        const ActivationFunctionPtr& getActivationFunc() const { return activation; }
        
    protected:
        void initWeights();
        
    protected:
        math::InputVector input;
        math::Vector output;
        math::Matrix weights;
        
        ActivationFunctionPtr activation;
    };
    
    typedef std::shared_ptr<Layer> LayerPtr;
}

#endif /* Layer_hpp */