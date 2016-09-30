//
//  Trainer.hpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Trainer_hpp
#define Trainer_hpp

#include "../AstraNet.hpp"
#include <vector>

namespace astra {
    
    class Trainer {
    public:
        explicit Trainer(AstraNetPtr& net);
        
        void train(double epsilon);
        void trainLayer(Layer* layer, double epsilon);
        
    protected:
        double epsilon;
        AstraNetPtr net;
    };
    
    typedef std::shared_ptr<Trainer> TrainerPtr;
    
}

#endif /* Trainer_hpp */
