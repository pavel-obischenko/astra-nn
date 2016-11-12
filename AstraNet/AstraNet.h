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
        static AstraNetPtr constructFullConnNet(unsigned int nInputs, const std::vector<int> &layerSizes);
        static AstraNetPtr constructFullConnSoftmaxNet(unsigned int nInputs, const std::vector<int> &layerSizes);

    public:
        std::vector<LayerPtr>& getLayers() { return layers; }
        void setLayers(const std::vector<LayerPtr>& layers) { this->layers = layers; }
        void addLayer(const LayerPtr& layer) { layers.push_back(layer); }
        
    public:
        const Output& process(const Input& input);

    public:
        const Input &getLastInput() const { return lastInput; }
        const Output &getLastOutput() const { return lastOutput; }
        
    protected:
        void setLastInput(const Input &lastInput) { AstraNet::lastInput = lastInput; }
        void setLastOutput(const Output &lastOutput) { AstraNet::lastOutput = lastOutput; }

    protected:
        std::vector<LayerPtr> layers;

    protected:
        Input lastInput;
        Output lastOutput;
    };
    
}

#endif /* AstraNet_hpp */
