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
    
    typedef std::shared_ptr<std::vector<double>> TrainDataInputPtr;
    typedef std::shared_ptr<std::vector<double>> TrainDataOutputPtr;
    
    struct TrainData {
        TrainData() : input(std::make_shared<std::vector<double>>()), output(std::make_shared<std::vector<double>>()) {}
        explicit TrainData(const TrainDataInputPtr& input, const TrainDataOutputPtr& output) : input(input), output(output) {}
        
        std::shared_ptr<std::vector<double>> input;
        std::shared_ptr<std::vector<double>> output;
    };
    
    
    typedef std::shared_ptr<TrainData> TrainDataPtr;
    typedef std::shared_ptr<std::vector<TrainDataPtr>> TrainDataArrayPtr;
}


#endif /* TrainData_h */
