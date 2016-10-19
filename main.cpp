//
//  main.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include <iostream>
#include <random>
#include <cmath>

#include "AstraNet/Common/Iterators.h"
#include "AstraNet/Math/Matrix.h"

#include "AstraNet/AstraNet.h"
#include "AstraNet/Layers/TanhLayer.h"
#include "AstraNet/Layers/SigLayer.h"
#include "AstraNet/Trainers/TrainData.h"
#include "AstraNet/Trainers/Trainer.h"
#include "AstraNet/Math/Vector.h"


#include "AstraNet/Algorithms/Image2Cols.h"

using namespace astra;
using namespace astra::math;

int main(int argc, const char * argv[]) {
    AstraNetPtr net = AstraNet::constructFeedForwardNet(2, {8, 4, 2, 4, 8, 1});
    
    int count = 20000;
    
    Vector posVec = {.5, .5};
    Vector negVec = {-.5, -.5};
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-.2, .2);
    auto rnd = std::bind(distribution, generator);
    
    TrainDataPtr trainData = TrainData::createPtr();
    
    for (int i = 0; i < count; i++) {
        bool outValue = rnd() > 0;
        Vector noiseVec = {rnd(), rnd()};
        
        Vector inputVec = outValue ? posVec : negVec;
        inputVec += noiseVec;
        
        Vector outVec = {outValue ? .5 : -.5};
        
        trainData->addTrainPair(*inputVec.get_data_storage(), *outVec.get_data_storage());
    }
    
    double error = 0;
    double epsilon = 0.5;
    
    TrainerPtr trainer = std::make_shared<Trainer>(net, trainData);
    for (int i = 0; i < count; i++) {
        trainer->runTrainEpoch(epsilon);
        
        TrainDataPairPtr pairPtr = trainData->currentPair();
        const std::vector<double>& input = *pairPtr->first;
        const std::vector<double>& dOut = *pairPtr->second
        ;
        auto out = net->process(input);

        int errCount = 500;
        bool lastIteration = i == count-1;
        if ((i > 0 && i % errCount == 0) || lastIteration) {
            error *= 100. / errCount;
            std::cout << (lastIteration ? i+1 : i) << " error: " << error << std::endl;
            
            error = 0;
        }
        else {
            error += fabs(trainer->errorFactor(Vector(out), Vector(dOut)).sum());
        }
        
        Vector inVec(input);
        Vector outVec(out);
        
        //std::cout << i << " " << inVec << " " << outVec << std::endl;
        
        //std::cout << i << " " << trainer->errorFactor(outVec, Vector(*data.output)) << std::endl;
    }
    return 0;
}
