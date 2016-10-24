//
// Created by Pavel on 23/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ConvLayer_hpp
#define ConvLayer_hpp

#include "Layer.h"

namespace astra {

    class ConvLayer : public Layer {
    public:
        ConvLayer(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long filterWidth, unsigned long filterHeight, unsigned long nFilters);

    public:
        virtual const math::Vector& process(const math::Vector& input);

    private:
        unsigned long width, height;
        unsigned long nChannels;
        unsigned long nFilters;
    };

};

#endif // ConvLayer_hpp