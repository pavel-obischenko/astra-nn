//
//  Trainer.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Trainer.hpp"
#include "TrainLayerWrapper.h"

namespace astra {
    
    Trainer::Trainer(AstraNetPtr& net, TrainDataArrayPtr& trainDataArray) : net(net), epsilon(.03), trainDataArray(trainDataArray), currentEpoch(0), layerWrappers(std::make_shared<TrainLayerWrapperArray>()) {
        initWrappers();
    }
    
    void Trainer::initWrappers() {
        auto layers = net->getLayers();
        std::for_each(layers.begin(), layers.end(), [this](LayerPtr& layer) {
            this->layerWrappers->emplace_back(layer, nullptr);
        });
    }
    
    void Trainer::runTrainEpoch(double epsilon) {
        TrainDataPtr currentTrainData = (*trainDataArray)[currentEpoch];
        Vector out = Vector(net->process(*currentTrainData->input));
        Vector dOut = Vector(*(currentTrainData->output));
        
        auto rbegin = layerWrappers->rbegin();
        for(auto layerWr = rbegin; layerWr <= layerWrappers->rend(); ++layerWr) {
            auto prevWr = layerWr > rbegin ? *(layerWr - 1) : nullptr;
            trainLayer(*layerWr, prevWr, Vector(out), dOut, epsilon);
        };
        
        ++currentEpoch;
    }
    
    void Trainer::trainLayer(TrainLayerWrapperPtr& currentWr, TrainLayerWrapperPtr& prevWr, const Vector& out, const Vector& dOut, double epsilon) {
        const Vector& errorGrad = prevWr != nullptr ? errorGradient(*(prevWr->weightGradient)) : errorGradient(out, dOut);
        const Vector& layerIn = currentWr->layer->getInput();
        Matrix weightGrad = weightGradient(layerIn, errorGrad, currentWr->layer);
        weightGrad *= epsilon;
        
    }
    
    double Trainer::errorSqr(const Vector& out, const Vector& train) {
        Vector err = out - train;
        err = err.mul_termwise(err);
        return err.sum();
    }
    
    Vector Trainer::errorGradient(const Vector& out, const Vector& train) {
        return 2 * (out - train);
    }
    
    Vector Trainer::errorGradient(const Matrix& prevWeightGradient) {
        std::vector<double> sumArray;
        
        std::vector<Vector> cols = prevWeightGradient.get_cols_const();
        cols = std::vector<Vector>(cols.begin(), cols.end() - 1);
        
        std::for_each(cols.begin(), cols.end(), [&sumArray](const Vector& col) {
            sumArray.push_back(col.sum());
        });
        
        return Vector(sumArray);
    }
    
    Matrix Trainer::weightGradient(const Vector& input, const Vector& errorGrad, LayerPtr layer) {
        InputVector inputVec(input);
        Matrix weights = layer->getWeights();
        
        auto currentErrorDerivative = errorGrad.get_storage_const().begin();
        std::vector<Vector> resultDelta;
        
        std::for_each(weights.get_rows().begin(), weights.get_rows().end(), [&inputVec, layer, &resultDelta, &currentErrorDerivative](const Vector& rowWeights) {
            ActivationFunctionPtr activation = layer->getActivationFunc();
            
            double rowSum = inputVec.mul_termwise(rowWeights).sum();
            double derivative = activation->derivativeValue(rowSum);
            
            Vector dw = inputVec.mul_termwise(rowWeights * derivative) * (*currentErrorDerivative++);
            resultDelta.push_back(dw);
        });
        
        return Matrix(resultDelta);
    }
    
}
