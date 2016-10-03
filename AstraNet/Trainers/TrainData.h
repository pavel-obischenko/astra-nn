//
//  TrainData.h
//  astra-nn
//
//  Created by Pavel on 03/10/16.
//  Copyright © 2016 Pavel. All rights reserved.
//

#ifndef TrainData_h
#define TrainData_h

#include <vector>

namespace astra {
    
    struct TrainData {
        std::shared_ptr<std::vector<double>> input;
        std::shared_ptr<std::vector<double>> output;
    };
    
    typedef std::shared_ptr<TrainData> TrainDataPtr;
    typedef std::shared_ptr<std::vector<TrainDataPtr>> TrainDataArrayPtr;
}


#endif /* TrainData_h */
