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
#include "TrainLayerWrapper.h"

#include "../Math/Vector.h"
#include "../Math/Matrix.h"

#include <vector>

namespace astra {
    
    typedef std::vector<double> Error;
    
    class Trainer {
    public:
        explicit Trainer(const AstraNetPtr& netPtr, const TrainDataPtr& trainDataPtr);
        void runTrainEpoch(double epsilon);

        void initWrappers();
        void trainLayer(TrainLayerWrapperPtr& currentWr, TrainLayerWrapperPtr& prevWr, const astra::math::Vector& out, const astra::math::Vector& dOut, double epsilon);
        
        double errorSqr(const astra::math::Vector& out, const astra::math::Vector& train);
        astra::math::Vector errorFactor(const astra::math::Vector& out, const astra::math::Vector& train);
        astra::math::Vector errorFactor(const astra::math::Matrix& prevWeights, const astra::math::Vector& prevLocalGradient);
        astra::math::Vector localGradient(const astra::math::InputVector& input, const astra::math::Vector& errorFactor, const astra::math::Matrix& weights, const ActivationFunctionPtr& activation);
        astra::math::Matrix calculateCorrectWeights(const astra::math::Matrix& weights, const astra::math::InputVector& input, const astra::math::Vector& localGrad, double epsilon);
        
    protected:
        double epsilon;
        
        AstraNetPtr netPtr;
        TrainDataPtr trainDataPtr;
        TrainLayerWrapperArrayPtr layerWrappers;
        
    protected:
        unsigned int currentEpoch;
    };
    
    typedef std::shared_ptr<Trainer> TrainerPtr;
    
}

#endif /* Trainer_hpp */
