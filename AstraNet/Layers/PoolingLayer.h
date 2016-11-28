//
// Created by Pavel Obischenko on 11/11/2016.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_POOLINGLAYER_H
#define ASTRA_NN_POOLINGLAYER_H

#include "Layer.h"
#include "PoolingLayerPtr.h"

namespace astra {

    enum PoolingLayerType {
        kPoolingLayerTypeMax,
        kPoolingLayerTypeAverage
    };

    class PoolingLayer : public Layer {
    public:
        static LayerPtr createAveragePoolingLayerPtr(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long poolWidth, unsigned long poolHeight);
        static LayerPtr createMaxPoolingLayerPtr(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long poolWidth, unsigned long poolHeight);
        static LayerPtr createPoolingLayerPtr(PoolingLayerType type, unsigned long width, unsigned long height, unsigned long nChannels, unsigned long poolWidth, unsigned long poolHeight);

    public:
        PoolingLayer(PoolingLayerType type, unsigned long width, unsigned long height, unsigned long nChannels, unsigned long poolingWidth, unsigned long poolingHeight);

    public:
        virtual LayerTrainerPtr createTrainer();
        virtual const math::Vector& process(const math::Vector& input);

    public:
        unsigned long getWidth() const { return width; }
        unsigned long getHeight() const { return height; }
        unsigned long getNChannels() const { return nChannels; }
        unsigned long getPoolingWidth() const { return poolingWidth; }
        unsigned long getPoolingHeight() const { return poolingHeight; }
        PoolingLayerType getType() const { return type; };

    private:
        unsigned long width, height;
        unsigned long nChannels;
        unsigned long poolingWidth, poolingHeight;
        PoolingLayerType type;
    };
}


#endif //ASTRA_NN_POOLINGLAYER_H
