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

#include "AstraNet/Common/Iterators.hpp"
#include "AstraNet/Math/Matrix.hpp"


#include "AstraNet/AstraNet.hpp"
#include "AstraNet/Layers/TanhLayer.hpp"
#include "AstraNet/Layers/SigLayer.hpp"
#include "AstraNet/Trainers/TrainData.h"
#include "AstraNet/Trainers/Trainer.hpp"
#include "AstraNet/Math/Vector.hpp"


using namespace astra;

int main(int argc, const char * argv[]) {
//    Vector v = {1, 2, 3, 4, 5, 6};
//    Vector h = v.head(5);
//    Vector t = v.tail(1);
//    
//    std::cout << v << std::endl;
//    std::cout << h << std::endl;
//    std::cout << t << std::endl;
    
    astra::math::Matrix m0 = {{1, 1, 1, 10, 1, 1, 10, 1},
                              {1, 2, 2, 20, 2, 2, 20, 1},
                              {1, 2, 3, 30, 3, 3, 20, 1},
                              {1, 2, 3, 4, 4, 3, 2, 1},
                              {1, 2, 3, 3, 3, 3, 2, 1},
                              {1, 2, 2, 2, 2, 2, 2, 1},
                              {1, 1, 1, 1, 1, 1, 1, 1}};
    
//    m0.for_each_col([](const astra::math::MatrixPtr& col) {
//        std::cout << *col << std::endl;
//    });
    
    astra::math::MatrixPtr sm1 = m0.submatrix(0, 0, 3, 2);
    astra::math::MatrixPtr sm2 = m0.submatrix(4, 0, 2, 3);
    
    std::cout << *sm1 << std::endl;
    std::cout << *sm2 << std::endl;
    std::cout << *sm1 * *sm2 << std::endl;
    
    //std::cout << m0 << std::endl;
    
    return 0;
    
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
        
        trainData->addTrainPair(inputVec.get_storage(), outVec.get_storage());
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
