//
// Created by Pavel on 03/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "TrainDataGenerator.h"
#include "../Math/Math.h"

#include <random>
#include <cmath>

using namespace astra::math;

namespace astra {

    TrainDataPtr TrainDataGenerator::xSinXRegression(unsigned long count, double minX/* = -1.*/, double maxX/* = 1.*/, double alpha/* = 1.*/, double betha/* = 8.*/, double noise/* = 0*/) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-noise, noise);
        auto rnd = std::bind(distribution, generator);

        TrainDataPtr trainData = TrainData::createPtr();

        double step = (maxX - minX) / count;
        double x = minX;

        for (unsigned long i = 0; i < count; ++i, x += step) {
            Vector inputVec = {x};
            if (noise > 0) {
                Vector noiseVec = {rnd()};
                inputVec += noiseVec;
            }

            Vector outVec = {alpha * x * std::sin(betha * x)};
            trainData->addTrainPair(*inputVec.get_data_storage(), *outVec.get_data_storage());
        }
        return trainData;
    }

    TrainDataPtr TrainDataGenerator::twoClassesClassification(unsigned long count) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-.2, .2);
        auto rnd = std::bind(distribution, generator);

        TrainDataPtr trainData = TrainData::createPtr();

        Vector posVec = {.5, .5};
        Vector negVec = {-.5, -.5};

        for (unsigned long i = 0; i < count; i++) {
            bool outValue = rnd() > 0;
            Vector noiseVec = {rnd(), rnd()};

            Vector inputVec = outValue ? posVec : negVec;
            inputVec += noiseVec;

            Vector outVec = outValue ? Vector({1., -1.}) : Vector({-1., 1.});
            trainData->addTrainPair(*inputVec.get_data_storage(), *outVec.get_data_storage());
        }
        return trainData;
    }

    TrainDataPtr TrainDataGenerator::threeClassesClassification(unsigned long count) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-.2, .2);
        auto rnd = std::bind(distribution, generator);

        TrainDataPtr trainData = TrainData::createPtr();

        Vector posVec = {.5, .5};
        Vector negVec = {-.5, -.5};
        Vector zeroVec = {0, 0};

        for (unsigned long i = 0; i < count; i++) {
            int outValue = (int)(std::fabs(rnd()) * 100)  % 3;
            Vector noiseVec = {rnd(), rnd()};
            Vector inputVec = zeroVec;
            Vector outVec = {0, 1., 0};

            switch (outValue) {
                case 1:
                    inputVec = posVec;
                    outVec = {0, 0, 1.};
                    break;
                case 2:
                    inputVec = negVec;
                    outVec = {1., 0, 0};
                    break;
                default:
                    break;
            }

            inputVec += noiseVec;
            trainData->addTrainPair(*inputVec.get_data_storage(), *outVec.get_data_storage());
        }
        return trainData;
    }
}
