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
        explicit Trainer(AstraNetPtr& net, TrainDataArrayPtr& trainDataVec);
        void runTrainEpoch(double epsilon);
        
    protected:
        void initWrappers();
        void trainLayer(TrainLayerWrapperPtr& currentWr, TrainLayerWrapperPtr& prevWr, const Vector& out, const Vector& dOut, double epsilon);
        
        double errorSqr(const Vector& out, const Vector& train);
        Vector errorGradient(const Vector& out, const Vector& train);
        Vector errorGradient(const Matrix& prevWeightGradient);
        Matrix weightGradient(const Vector& input, const Vector& errorGrad, LayerPtr layer);
        
    protected:
        double epsilon;
        AstraNetPtr net;
        TrainDataArrayPtr trainDataArray;
        TrainLayerWrapperArrayPtr layerWrappers;
        
    protected:
        unsigned int currentEpoch;
    };
    
    typedef std::shared_ptr<Trainer> TrainerPtr;
    
}

#endif /* Trainer_hpp */
