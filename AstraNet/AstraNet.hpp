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
        
    protected:
        std::vector<std::shared_ptr<Layer>> layers;
    };
    
}

#endif /* AstraNet_hpp */
