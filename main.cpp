//
//  main.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include <iostream>
#include <random>

#include "AstraNet/AstraNet.hpp"
#include "AstraNet/Layers/TanhLayer.hpp"
#include "AstraNet/Layers/SigLayer.hpp"
#include "AstraNet/Trainers/TrainData.h"
#include "AstraNet/Trainers/Trainer.hpp"
#include "AstraNet/Math/Vector.hpp"

using namespace astra;

int main(int argc, const char * argv[]) {
    AstraNetPtr net = std::make_shared<AstraNet>();
    
    LayerPtr layerFirst = std::make_shared<TanhLayer>(2, 5, 1.);
    net->addLayer(layerFirst);
    LayerPtr layerLast = std::make_shared<TanhLayer>(5, 1, 1.);
    net->addLayer(layerLast);
    
    int count = 3000;
    
    Vector posVec = {.5, .5};
    Vector negVec = {-.5, -.5};
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-.2, .2);
    auto rnd = std::bind(distribution, generator);
    
    TrainDataArrayPtr trainArray = std::make_shared<std::vector<TrainDataPtr>>();
    
    for (int i = 0; i < count; i++) {
        bool outValue = rnd() > 0;
        Vector noiseVec = {rnd(), rnd()};
        
        Vector inputVec = outValue ? posVec : negVec;
        inputVec += noiseVec;
        
        Vector outVec = {outValue ? .5 : -.5};
        
        TrainDataInputPtr in = std::make_shared<std::vector<double>>(inputVec.get_storage());
        TrainDataInputPtr out = std::make_shared<std::vector<double>>(outVec.get_storage());
        
        TrainDataPtr trainData = std::make_shared<TrainData>(in, out);
        trainArray->push_back(trainData);
    }
    
    TrainerPtr trainer = std::make_shared<Trainer>(net, trainArray);
    for (int i = 0; i < count; i++) {
        const TrainData& data = (*(*trainArray)[i]);
        const std::vector<double>& input = *data.input;
        auto out = net->process(input);
        
        if (i > 0 && i % 100 == 0) {
            int i = 0;
        }
        
        Vector inVec(input);
        Vector outVec(out);
        
        std::cout << i << " " << inVec << " " << outVec << std::endl;
        
        trainer->runTrainEpoch(0.05);
        
        //std::cout << i << " " << trainer->errorFactor(outVec, Vector(*data.output)) << std::endl;
    }
    return 0;
}
