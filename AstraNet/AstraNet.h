//
//  AstraNet.hpp
//  astra-nn
//
//  Created by Pavel on 29/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef AstraNet_hpp
#define AstraNet_hpp

#include "Layers/Layer.h"
#include <vector>

namespace astra {
    
    typedef std::vector<double> Input;
    typedef std::vector<double> Output;
    
    class AstraNet;
    typedef std::shared_ptr<AstraNet> AstraNetPtr;
    
    class AstraNet {
    public:
        static AstraNetPtr createPtr();
        static AstraNetPtr constructFeedForwardNet(unsigned int nInputs, const std::vector<int>& layerSizes);
        
    public:
        
        Output process(const Input& input);
        
        std::vector<LayerPtr>& getLayers() { return layers; }
        void setLayers(const std::vector<LayerPtr>& layers) { this->layers = layers; }
        void addLayer(const LayerPtr& layer) {
            layers.push_back(layer);
        }
        
    protected:
        std::vector<LayerPtr> layers;
    };
    
}

#endif /* AstraNet_hpp */
