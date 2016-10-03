//
//  Trainer.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Trainer.hpp"

namespace astra {
    
    static double errorSqr(const Output& out, const Output& train);
    
    Trainer::Trainer(AstraNetPtr net, TrainDataArrayPtr trainDataArray) : net(net), epsilon(.03), trainDataArray(trainDataArray), currentEpoch(0) {}
    
    void Trainer::train(double epsilon) {
        TrainDataPtr currentTrainData = (*trainDataArray)[currentEpoch];
        
        Output& d = *currentTrainData->output;
        Output y = net->process(*currentTrainData->input);
        
        //double err = errorSqr(y, d);
        
        ++currentEpoch;
    }
    
    void Trainer::trainLayer(Layer* currentLayer, Layer* prevLayer, double epsilon, const Error& error) {
        
    }
    
    static double errorSqr(const Vector& out, const Vector& train) {
        Vector err = out - train;
        err = err.mul_termwise(err);
        return err.sum();
    }
    
    static Vector errorGradient(const Vector& out, const Vector& train) {
        return 2 * (out - train);
    }
    
    static Matrix weightGradient(const Vector& input, const Vector& errorGrad, Layer* layer) {
        InputVector inputVec(input);
        Matrix weights = layer->getWeights();
        
        auto currentErrorDerivative = errorGrad.get_storage_const().begin();
        std::vector<Vector> resultDelta;
        
        std::for_each(weights.get_rows().begin(), weights.get_rows().end(), [&inputVec, layer, &resultDelta, &currentErrorDerivative](const Vector& rowWeights) {
            ActivationFunctionPtr activation = layer->getActivationFunc();
            
            double sum = inputVec.mul_termwise(rowWeights).sum();
            double derivative = activation->derivativeValue(sum);
            
            Vector dw = inputVec.mul_termwise(rowWeights * derivative) * (*currentErrorDerivative++);
            resultDelta.push_back(dw);
        });
        
        return Matrix(resultDelta);
    }
    
}
