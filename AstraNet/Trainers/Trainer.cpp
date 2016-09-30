//
//  Trainer.cpp
//  astra-nn
//
//  Created by Pavel on 30/09/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#include "Trainer.hpp"

namespace astra {
    
    Trainer::Trainer(AstraNetPtr& net) : net(net), epsilon(.03) {}
    
    void Trainer::train(double epsilon) {
        
    }
    
    void Trainer::trainLayer(Layer* layer, double epsilon) {
        
    }
    
}
