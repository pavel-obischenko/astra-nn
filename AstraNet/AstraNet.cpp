//
//  AstraNet.cpp
//  astra-nn
//
//  Created by Pavel on 29/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "AstraNet.h"
#include "Layers/FullConnLayer.h"
#include "Math/Vector.h"

using namespace astra::math;

namespace astra {
    
    AstraNetPtr AstraNet::createPtr() {
        return std::make_shared<AstraNet>();
    }
    
    AstraNetPtr AstraNet::constructFullConnNet(unsigned int nInputs, const std::vector<int> &layerSizes) {
        AstraNetPtr netPtr = AstraNet::createPtr();
        
        int currentInputsCount = nInputs;
        std::for_each(layerSizes.begin(), layerSizes.end(), [&netPtr, &currentInputsCount](int layerSize) {
            LayerPtr layerPtr = FullConnLayer::createTanhLayerPtr(currentInputsCount, layerSize, 1.);
            netPtr->addLayer(layerPtr);
            
            currentInputsCount = layerSize;
        });
                
        return netPtr;
    }
    
    Output AstraNet::process(const Input& input) {
        auto firstLayer = layers.begin();
        
        Vector lastOutput;
        for (auto layer = firstLayer; layer != layers.end(); ++layer) {
            const Vector& currentInput = layer == firstLayer ? Vector(input) : lastOutput;
            lastOutput = (*layer)->process(currentInput);
        }
        
        return *lastOutput.get_data_storage();
    }
}
