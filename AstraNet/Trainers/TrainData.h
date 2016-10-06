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
#include <random>
#include <iterator>

namespace astra {
    
    typedef std::shared_ptr<std::vector<double>> TrainDataInputPtr;
    typedef std::shared_ptr<std::vector<double>> TrainDataOutputPtr;
    
    typedef std::pair<TrainDataInputPtr, TrainDataOutputPtr> TrainDataPair;
    typedef std::shared_ptr<TrainDataPair> TrainDataPairPtr;
    
    typedef std::vector<TrainDataPairPtr> TrainDataVector;
    typedef std::shared_ptr<TrainDataVector> TrainDataVectorPtr;
    
    
    static TrainDataVectorPtr createTrainDataVectorPtr() {
        return std::make_shared<TrainDataVector>();
    }
    
    static TrainDataPairPtr createTrainDataPair(const std::vector<double>& input, const std::vector<double>& output) {
        TrainDataInputPtr inputPtr = std::make_shared<std::vector<double>>(input);
        TrainDataInputPtr outputPtr = std::make_shared<std::vector<double>>(output);
        
        return std::make_shared<TrainDataPair>(inputPtr, outputPtr);
    }
    
    class TrainData;
    typedef std::shared_ptr<TrainData> TrainDataPtr;
    
    class TrainData {
    public:
        static TrainDataPtr createPtr() {
            return std::make_shared<TrainData>();
        }
        
        TrainData() : trainDataVectorPtr(createTrainDataVectorPtr()), mt(rd()) {}
        
        void addTrainPair(const std::vector<double>& input, const std::vector<double>& output) {
            trainDataVectorPtr->push_back(createTrainDataPair(input, output));
        }
        
        const TrainDataPairPtr& nextPair() {
            unsigned long max = trainDataVectorPtr->size() - 1;
            int index = std::uniform_int_distribution<int>{0, static_cast<int>(max)}(mt);
            
            currentDataPair = trainDataVectorPtr->at(index);
            
            return currentPair();
        }
        
        const TrainDataPairPtr& currentPair() {
            return currentDataPair;
        }
        
    private:
        TrainDataVectorPtr trainDataVectorPtr;
        TrainDataPairPtr currentDataPair;
        
        std::random_device rd;
        std::mt19937 mt;
    };
}


#endif /* TrainData_h */
