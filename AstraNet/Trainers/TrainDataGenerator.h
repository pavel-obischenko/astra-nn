//
// Created by Pavel on 03/11/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_TRAINDATAGENERATOR_H
#define ASTRA_NN_TRAINDATAGENERATOR_H

#include "TrainData.h"

namespace astra {

    class TrainDataGenerator {
    public:
        static TrainDataPtr xSinXRegression(unsigned long count, double minX = -1., double maxX = 1., double alpha = 1., double betha = 8., double noise = 0);
        static TrainDataPtr twoClassesClassification(unsigned long count);
        static TrainDataPtr threeClassesClassification(unsigned long count);
        static TrainDataPtr twoLinesClassesClassification(unsigned long count);

    };
}


#endif //ASTRA_NN_TRAINDATAGENERATOR_H
