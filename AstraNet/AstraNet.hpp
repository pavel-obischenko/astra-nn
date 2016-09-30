//
//  AstraNet.hpp
//  astra-nn
//
//  Created by Pavel on 29/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef AstraNet_hpp
#define AstraNet_hpp

#include "Layers/Layer.hpp"
#include <vector>

namespace astra {
    
    typedef std::vector<double> Input;
    typedef std::vector<double> Output;
    
    class AstraNet {
    public:
        Output process(const Input& input);
        
        std::vector<LayerPtr>& getLayers() { return layers; }
        void setLayers(const std::vector<LayerPtr>& layers) { this->layers = layers; }
        
    protected:
        std::vector<LayerPtr> layers;
    };
    
    typedef std::shared_ptr<AstraNet> AstraNetPtr;
    
}

#endif /* AstraNet_hpp */
