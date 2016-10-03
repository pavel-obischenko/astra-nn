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
#include "TrainData.h"

#include <vector>

namespace astra {
    
    typedef std::vector<double> Error;
    
    class Trainer {
    public:
        explicit Trainer(AstraNetPtr net, TrainDataArrayPtr trainDataVec);
        
        void train(double epsilon);
        void trainLayer(Layer* currentLayer, Layer* prevLayer, double epsilon, const Error& error);
        
    protected:
        double epsilon;
        AstraNetPtr net;
        TrainDataArrayPtr trainDataArray;
        
    protected:
        unsigned int currentEpoch;
    };
    
    typedef std::shared_ptr<Trainer> TrainerPtr;
    
}

#endif /* Trainer_hpp */
