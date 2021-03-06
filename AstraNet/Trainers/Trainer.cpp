//
//  Trainer.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Trainer.h"
#include "LayerTrainer.h"

using namespace astra::math;

namespace astra {
    
    Trainer::Trainer(const AstraNetPtr& netPtr, const TrainDataPtr& trainDataPtr) : netPtr(netPtr), epsilon(.03), trainDataPtr(trainDataPtr), currentEpoch(0), trainers() {
        init();
    }
    
    void Trainer::init() {
        auto layers = netPtr->getLayers();
        std::for_each(layers.begin(), layers.end(), [this](LayerPtr& layer) {
            this->trainers.push_back(layer->createTrainer());
        });
    }
    
    Vector Trainer::runTrainEpoch(double epsilon, double momentum) {
        TrainDataPairPtr currentTrainData = trainDataPtr->nextPair();
        Vector out = Vector(netPtr->process(*currentTrainData->first));
        Vector dOut = Vector(*(currentTrainData->second));

        Vector error = errorFactor(out, dOut);
        Vector lastError = error;
        for(auto trainer = trainers.rbegin(); trainer != trainers.rend(); ++trainer) {
            lastError = (*trainer)->backpropagateError(lastError, epsilon, momentum);
        };
        
        ++currentEpoch;
        return error;
    }
    
    Vector Trainer::errorFactor(const Vector& out, const Vector& dOut) {
        auto layerTrainer = trainers.rbegin();
        return  (*layerTrainer)->trainingError(out, dOut);
    }
}
