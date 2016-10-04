//
//  AstraNet.cpp
//  astra-nn
//
//  Created by Pavel on 29/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "AstraNet.hpp"

namespace astra {
    
    Output AstraNet::process(const Input& input) {
        auto firstLayer = layers.begin();
        
        vec lastOutput;
        for (auto layer = firstLayer; layer != layers.end(); ++layer) {
            const vec& currentInput = layer == firstLayer ? vec(input) : lastOutput;
            lastOutput = layer->get()->process(currentInput);
        }
        
        return lastOutput.get_storage_const();
    }
}
