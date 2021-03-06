//
//  AstraNet.cpp
//  astra-nn
//
//  Created by Pavel on 29/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "AstraNet.h"
#include "Layers/FullConnLayer.h"
#include "Layers/SoftmaxLayer.h"
#include "Math/Math.h"

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

    AstraNetPtr AstraNet::constructFullConnSoftmaxNet(unsigned int nInputs, const std::vector<int> &layerSizes) {
        AstraNetPtr netPtr = AstraNet::createPtr();

        int currentInputsCount = nInputs;
        auto lastLayer = layerSizes.end() - 1;
        std::for_each(layerSizes.begin(), lastLayer, [&netPtr, &currentInputsCount](int layerSize) {
            LayerPtr layerPtr = FullConnLayer::createTanhLayerPtr(currentInputsCount, layerSize, 1.);
            netPtr->addLayer(layerPtr);

            currentInputsCount = layerSize;
        });

        LayerPtr layerPtr = SoftmaxLayer::createSoftmaxLayerPtr(currentInputsCount, *lastLayer);
        netPtr->addLayer(layerPtr);

        return netPtr;
    }

    const Output& AstraNet::process(const Input& input) {
        setLastInput(input);
        auto firstLayer = layers.begin();
        
        Vector lastOutput;
        for (auto layer = firstLayer; layer != layers.end(); ++layer) {
            const Vector& currentInput = layer == firstLayer ? Vector(input) : lastOutput;
            lastOutput = (*layer)->process(currentInput);
        }

        setLastOutput(*lastOutput.get_data_storage());
        return getLastOutput();
    }
}
