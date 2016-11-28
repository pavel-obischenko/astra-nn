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

#include "AstraNet/Layers/SoftmaxLayer.h"
#include "AstraNet/Layers/FullConnLayer.h"
#include "AstraNet/Layers/ConvLayer.h"
#include "AstraNet/Layers/ReLULayer.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace astra;
using namespace astra::math;
using namespace astra::algorithms;

void threeClassesClassification();
void twoLinesClassesClassification();
void xSinXregression();

int main(int argc, const char * argv[]) {
    twoLinesClassesClassification();
    //threeClassesClassification();
    //xSinXregression();
    return 0;
}

void twoLinesClassesClassification() {
    AstraNetPtr netPtr = AstraNet::createPtr();

    int width = 6;
    int height = 6;
    int channels = 1;
    int filterWidth = 3;
    int filterHeight = 3;
    int filters = 2;
    LayerPtr convLayer = ConvLayer::createConvLayerPtr(width, height, channels, filterWidth, filterHeight, filters);
    netPtr->addLayer(convLayer);

    int reluSize = (int)convLayer->getOutput().size();
    LayerPtr reluLayerPtr = ReLULayer::createReLULayerPtr(reluSize);
    netPtr->addLayer(reluLayerPtr);

    int softmaxSize = 2;
    LayerPtr softmaxLayerPtr = SoftmaxLayer::createSoftmaxLayerPtr(reluLayerPtr->getOutput().size(), softmaxSize);
    netPtr->addLayer(softmaxLayerPtr);

    int count = 64;
    int epochCount = 50000;

    TrainDataPtr trainData = TrainDataGenerator::twoLinesClassesClassification(count);

    double errorSum = 0;
    double epsilon = 0.1;
    double momentum = 0.3;

    double minError = 999999999999.;

    TrainerPtr trainer = std::make_shared<Trainer>(netPtr, trainData);
    for (int i = 0; i < epochCount; i++) {
        double  err = trainer->runTrainEpoch(epsilon, momentum).length();
        errorSum += err;

        int errCount = 100;
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
}

void threeClassesClassification() {
    AstraNetPtr net = AstraNet::constructFullConnSoftmaxNet(2, {4, 3});

    int count = 1000;
    int epochCount = 50000;
    //TrainDataPtr trainData = TrainDataGenerator::firstRegression(count);
    //TrainDataPtr trainData = TrainDataGenerator::xSinXRegression(count, 0, 1, 1, 12, 0);
    //TrainDataPtr trainData = TrainDataGenerator::twoClassesClassification(count);
    TrainDataPtr trainData = TrainDataGenerator::threeClassesClassification(count);

    double errorSum = 0;
    double epsilon = 0.2;
    double momentum = 0.5;

    double minError = 999999999999.;

    TrainerPtr trainer = std::make_shared<Trainer>(net, trainData);
    for (int i = 0; i < epochCount; i++) {
        double  err = trainer->runTrainEpoch(epsilon, momentum).length();
        errorSum += err;

        int errCount = 100;
        bool lastIteration = i == epochCount-1;
        if ((i > 0 && i % errCount == 0) || lastIteration) {
            double err = errorSum / errCount;
            minError = std::min(err, minError);
            std::cout << (lastIteration ? i+1 : i) << " error: " << err << std::endl;
            errorSum = 0;

            if (err < 0.015) {
                std::cout << "training breaked" << std::endl;
                break;
            }
        }
    }
    std::cout << "min error: " << minError << std::endl;
}

void xSinXregression() {
    AstraNetPtr net = AstraNet::constructFullConnNet(1, {10, 10, 1});

    int count = 1000;
    int epochCount = 50000;
    TrainDataPtr trainData = TrainDataGenerator::xSinXRegression(count, 0, 1, 1, 12, 0);

    double errorSum = 0;
    double epsilon = 0.1;
    double momentum = 0.5;

    double minError = 999999999999.;

    TrainerPtr trainer = std::make_shared<Trainer>(net, trainData);
    for (int i = 0; i < epochCount; i++) {
        double  err = trainer->runTrainEpoch(epsilon, momentum).length();
        errorSum += err;

        int errCount = 100;
        bool lastIteration = i == epochCount-1;
        if ((i > 0 && i % errCount == 0) || lastIteration) {
            double err = errorSum / errCount;
            minError = std::min(err, minError);
            std::cout << (lastIteration ? i+1 : i) << " error: " << err << std::endl;
            errorSum = 0;

            if (err < 0.02) {
                std::cout << "training breaked" << std::endl;
                break;
            }
        }
    }

    std::cout << "min error: " << minError << std::endl;
    std::cout << "start visualisation..."<< std::endl;

    std::vector<double> xVec, yVec, tVec;

    int pointsCount = 200;
    double minX = 0;
    double maxX = 1.;
    double x = minX;
    double stride = (maxX - minX) / pointsCount;

    for (int i = 0; i < pointsCount; ++i, x += stride) {
        xVec.push_back(x);
        yVec.push_back(net->process({x}).at(0));
        tVec.push_back(x * std::sin(x * 12.));
    }

    plt::plot(xVec, yVec);
    plt::plot(xVec, tVec);
    plt::show();


}
