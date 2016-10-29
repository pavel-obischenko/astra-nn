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

#include "AstraNet/AstraNet.h"
#include "AstraNet/Layers/FullConnLayer.h"

#include "AstraNet/Trainers/TrainData.h"
#include "AstraNet/Trainers/Trainer.h"

#include "AstraNet/Math/Math.h"
#include "AstraNet/Algorithms/Image2Cols.h"

#include "AstraNet/ActivationFunctions/SigActivationFunction.h"

using namespace astra;
using namespace astra::math;

int main(int argc, const char * argv[]) {
//    Matrix src = {{1, 2, 3, 4, 5},
//                  {1, 2, 3, 4, 5},
//                  {1, 2, 3, 4, 5},
//                  {1, 2, 3, 4, 5},
//                  {1, 2, 3, 4, 5}};
//
//    Matrix src = {{1, 2, 2, 1},
//                  {2, 4, 4, 2},
//                  {2, 4, 4, 2},
//                  {1, 2, 2, 1}}; // патчи (ядра свертки) - столбцы
//
    Matrix f = {{0, 1, 0, 0, 0},
                {0, 0, 0, 1, 0}};   // фильтры (наборы весов) - строки

    Vector v = {0.1, 0.1, 1,
                0.1, 0.1, 0.1,
                0.1, 0.1, 1};
    Matrix inp = *astra::algorithms::Image2Cols::convertToCols(v, 1, 2, 2, true);

    // inp - ядра свертки в строку + bias weight. Сколько фильтров - столько и строк
    std::cout << inp << std::endl;

    auto res = f * inp;
    std::cout << res << std::endl;

    Vector pe_vec = {-0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5};
    Matrix pe_mat(pe_vec.get_data_storage(), f.get_width() - 1, f.get_height());

    ActivationFunctionPtr activation = std::make_shared<SigActivationFunction>(0.1);
    auto derivatives = activation->derivativeValue(res);
    std::cout << derivatives << std::endl;

    auto localGradient = derivatives.element_wise_mul(pe_mat);
    std::cout << localGradient << std::endl;

    auto newW = f + (localGradient * inp.transpose()) * 0.1;
    std::cout << newW << std::endl;

    auto errFactorCols =  f.transpose() * localGradient;
    std::cout << errFactorCols << std::endl;

    return 0;
    AstraNetPtr net = AstraNet::constructFullConnNet(2, {8, 4, 2, 4, 8, 1});
    
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
    }
    return 0;
}
