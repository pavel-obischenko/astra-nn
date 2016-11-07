//
//  main.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "AstraNet/Trainers/TrainDataGenerator.h"

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
    AstraNetPtr net = AstraNet::constructFullConnNet(1, {4, 4, 1});
    
    int count = 1000;
    int epochCount = 50000;
    //TrainDataPtr trainData = TrainDataGenerator::firstRegression(count);
    TrainDataPtr trainData = TrainDataGenerator::xSinXRegression(count, 0, 1, 1, 12, 0);
    //TrainDataPtr trainData = TrainDataGenerator::twoClassesClassification(count);
    
    double errorSum = 0;
    double epsilon = 0.15;
    double momentum = 0.5;

    double minError = 999999999999.;
    
    TrainerPtr trainer = std::make_shared<Trainer>(net, trainData);
    for (int i = 0; i < epochCount; i++) {
        double  err = trainer->runTrainEpoch(epsilon, momentum).length();
        errorSum += err;

        int errCount = 500;
        bool lastIteration = i == epochCount-1;
        if ((i > 0 && i % errCount == 0) || lastIteration) {
            double err = errorSum / errCount;
            minError = std::min(err, minError);
            std::cout << (lastIteration ? i+1 : i) << " error: " << err << std::endl;
            errorSum = 0;

            if (err < 0.05) {
                std::cout << "training breaked" << std::endl;
                break;
            }
        }
    }
    std::cout << "min error: " << minError << std::endl;

    return 0;
}
