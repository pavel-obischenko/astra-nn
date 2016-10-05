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
    
    Trainer::Trainer(AstraNetPtr& net, const TrainDataArrayPtr& trainDataArray) : net(net), epsilon(.03), trainDataArray(trainDataArray), currentEpoch(0), layerWrappers(std::make_shared<TrainLayerWrapperArray>()) {
        initWrappers();
    }
    
    void Trainer::initWrappers() {
        auto layers = net->getLayers();
        std::for_each(layers.begin(), layers.end(), [this](LayerPtr& layer) {
            this->layerWrappers->push_back(std::make_shared<TrainLayerWrapper>(layer, nullptr));
        });
    }
    
    void Trainer::runTrainEpoch(double epsilon) {
        TrainDataPtr currentTrainData = (*trainDataArray)[currentEpoch];
        Vector out = Vector(net->process(*currentTrainData->input));
        Vector dOut = Vector(*(currentTrainData->output));
        
        auto rbegin = layerWrappers->rbegin();
        for(auto layerWr = rbegin; layerWr != layerWrappers->rend(); ++layerWr) {
            auto prevWr = layerWr > rbegin ? *(layerWr - 1) : nullptr;
            trainLayer(*layerWr, prevWr, Vector(out), dOut, epsilon);
        };
        
        ++currentEpoch;
    }
    
    void Trainer::trainLayer(TrainLayerWrapperPtr& currentWr, TrainLayerWrapperPtr& prevWr, const Vector& out, const Vector& dOut, double epsilon) {
        const Vector& errFactor = prevWr != nullptr ? errorFactor(prevWr->layer->getWeights(), *(prevWr->localGradient)) : errorFactor(out, dOut);
        
        LayerPtr currentLayerPtr = currentWr->layer;
        const InputVector& layerIn = currentLayerPtr->getInput();
        Vector localGrad = localGradient(layerIn, errFactor, *currentLayerPtr);
        
        currentWr->localGradient = std::make_shared<Vector>(localGrad);
        
        Matrix correctWeights = calculateCorrectWeights(currentLayerPtr->getWeights(), layerIn, localGrad, epsilon);
        currentLayerPtr->setWeights(correctWeights);
    }
    
    double Trainer::errorSqr(const Vector& out, const Vector& train) {
        Vector err = errorFactor(out, train);
        err = err.mul_termwise(err);
        return err.sum();
    }
    
    Vector Trainer::errorFactor(const Vector& out, const Vector& train) {
        return train - out;
    }
    
    Vector Trainer::errorFactor(const Matrix& prevWeights, const Vector& prevLocalGradient) {
        std::vector<double> sumArray;
        
        std::vector<Vector> cols = prevWeights.get_cols_const();
        
        // col < cols.end() - without last col, last col is bias weight
        for (auto col = cols.begin(); col < cols.end(); ++col) {
            sumArray.push_back(col->mul_termwise(prevLocalGradient).sum());
        }
        
        std::for_each(cols.begin(), cols.end(), [&sumArray](const Vector& col) {
            
            sumArray.push_back(col.sum());
        });
        
        return Vector(sumArray);
    }
    
    Vector Trainer::localGradient(const InputVector& input, const Vector& errorFactor, const Layer& layer) {
        Matrix weights = layer.getWeights();
        
        auto currentErrorFactor = errorFactor.get_storage_const().begin();
        std::vector<double> resultDelta;
        
        std::for_each(weights.get_rows().begin(), weights.get_rows().end(), [&input, &layer, &resultDelta, &currentErrorFactor](const Vector& rowWeights) {
            ActivationFunctionPtr activation = layer.getActivationFunc();
            
            double rowSum = input.mul_termwise(rowWeights).sum();
            double derivative = activation->derivativeValue(rowSum);
            
            double grad = derivative * (*currentErrorFactor++);
            resultDelta.push_back(grad);
        });
        
        return Vector(resultDelta);
    }
    
    Matrix Trainer::calculateCorrectWeights(const Matrix& weights, const Vector& input, const Vector& localGrad, double epsilon) {
        std::vector<Vector> resultData;
        
        //std::cout << "w:" << weights << " lg:" << localGrad << std::endl;
        
        auto currentLocalGrad = localGrad.get_storage_const().begin();
        
        std::for_each(weights.get_rows_const().begin(), weights.get_rows_const().end(), [&input, &currentLocalGrad, epsilon, &resultData](const Vector& rowWeights) {
            //Vector v0 = rowWeights.mul_termwise(input);
            Vector v1 = (*currentLocalGrad) * input;
            //Vector v1 = (*currentLocalGrad) * v0;
            Vector dVec = v1 * epsilon;
            //Vector dVec = input * (*currentLocalGrad) * epsilon;
            resultData.push_back(rowWeights + dVec);
            ++currentLocalGrad;
        });
        
        //std::cout << " new:" << Matrix(resultData) << std::endl;
        
        return Matrix(resultData);
    }
}
