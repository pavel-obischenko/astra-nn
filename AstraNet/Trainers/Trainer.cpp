//
//  Trainer.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Trainer.h"

using namespace astra::math;

namespace astra {
    
    Trainer::Trainer(const AstraNetPtr& netPtr, const TrainDataPtr& trainDataPtr) : netPtr(netPtr), epsilon(.03), trainDataPtr(trainDataPtr), currentEpoch(0), layerWrappers(std::make_shared<TrainLayerWrapperArray>()) {
        initWrappers();
    }
    
    void Trainer::initWrappers() {
        auto layers = netPtr->getLayers();
        std::for_each(layers.begin(), layers.end(), [this](LayerPtr& layer) {
            this->layerWrappers->push_back(std::make_shared<TrainLayerWrapper>(layer, nullptr));
        });
    }
    
    void Trainer::runTrainEpoch(double epsilon) {
        TrainDataPairPtr currentTrainData = trainDataPtr->nextPair();
        Vector out = Vector(netPtr->process(*currentTrainData->first));
        Vector dOut = Vector(*(currentTrainData->second));
        
        auto rbegin = layerWrappers->rbegin();
        for(auto layerWr = rbegin; layerWr != layerWrappers->rend(); ++layerWr) {
            auto prevWr = layerWr > rbegin ? *(layerWr - 1) : nullptr;
            trainLayer(*layerWr, prevWr, Vector(out), dOut, epsilon);
        };
        
        for(auto layerWr = rbegin; layerWr != layerWrappers->rend(); ++layerWr) {
            LayerPtr layer = (*layerWr)->layer;
            layer->setWeights(*(*layerWr)->newWeights);
        };
        
        ++currentEpoch;
    }
    
    void Trainer::trainLayer(TrainLayerWrapperPtr& currentWr, TrainLayerWrapperPtr& prevWr, const Vector& out, const Vector& dOut, double epsilon) {
        const Vector& errFactor = prevWr != nullptr ? errorFactor(prevWr->layer->getWeights(), *(prevWr->localGradient)) : errorFactor(out, dOut);
        
        LayerPtr currentLayerPtr = currentWr->layer;
        const InputVector& layerIn = currentLayerPtr->getInput();
        Vector localGrad = localGradient(layerIn, errFactor, currentLayerPtr->getWeights(), currentLayerPtr->getActivationFunc());
        
        currentWr->localGradient = std::make_shared<Vector>(localGrad);
        currentWr->newWeights = std::make_shared<Matrix>(calculateCorrectWeights(currentLayerPtr->getWeights(), layerIn, localGrad, epsilon));
    }
    
    double Trainer::errorSqr(const Vector& out, const Vector& train) {
        Vector err = errorFactor(out, train);
        err = err.element_wise_mul(err);
        return err.sum();
    }
    
    Vector Trainer::errorFactor(const Vector& out, const Vector& train) {
        return train - out;
    }
    
    Vector Trainer::errorFactor(const Matrix& prevWeights, const Vector& prevLocalGradient) {
        return *(prevWeights.transpose() * prevLocalGradient).head(prevWeights.get_width() - 1);
    }
    
    Vector Trainer::localGradient(const InputVector& input, const Vector& errorFactor, const Matrix& weights, const ActivationFunctionPtr& activation) {
        Vector derivative = activation->derivativeValue(weights * input.toVector());
        return derivative.element_wise_mul(errorFactor);
    }
    
    Matrix Trainer::calculateCorrectWeights(const Matrix& weights, const InputVector& input, const Vector& localGrad, double epsilon) {
        return weights + ((input * epsilon) * localGrad);
    }
}
