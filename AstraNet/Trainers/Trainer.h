//
//  Trainer.hpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef Trainer_hpp
#define Trainer_hpp

#include "../AstraNet.h"

#include "TrainData.h"
#include "LayerTrainerPtr.h"

#include "../Math/Vector.h"
#include "../Math/Matrix.h"

#include <vector>

namespace astra {
    
    typedef std::vector<double> Error;
    
    class Trainer {
    public:
        explicit Trainer(const AstraNetPtr& netPtr, const TrainDataPtr& trainDataPtr);

    public:
        void runTrainEpoch(double epsilon);
        astra::math::Vector errorFactor(const astra::math::Vector& out, const astra::math::Vector& dOut);

    protected:
        void init();
        
    protected:
        double epsilon;
        
        AstraNetPtr netPtr;
        TrainDataPtr trainDataPtr;
        LayerTrainerArray trainers;
        
    protected:
        unsigned int currentEpoch;
    };
    
    typedef std::shared_ptr<Trainer> TrainerPtr;
    
}

#endif /* Trainer_hpp */
