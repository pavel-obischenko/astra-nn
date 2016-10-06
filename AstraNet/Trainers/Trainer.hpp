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
#include "TrainLayerWrapper.h"

#include <vector>

namespace astra {
    
    typedef std::vector<double> Error;
    
    class Trainer {
    public:
        explicit Trainer(const AstraNetPtr& netPtr, const TrainDataPtr& trainDataPtr);
        void runTrainEpoch(double epsilon);
        
    //protected:
        void initWrappers();
        void trainLayer(TrainLayerWrapperPtr& currentWr, TrainLayerWrapperPtr& prevWr, const Vector& out, const Vector& dOut, double epsilon);
        
        double errorSqr(const Vector& out, const Vector& train);
        Vector errorFactor(const Vector& out, const Vector& train);
        Vector errorFactor(const Matrix& prevWeights, const Vector& prevLocalGradient);
        Vector localGradient(const InputVector& input, const Vector& errorGrad, const Layer& layer);
        Matrix calculateCorrectWeights(const Matrix& weights, const Vector& input, const Vector& localGrad, double epsilon);
        
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
