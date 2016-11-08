//
// Created by Pavel on 23/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ConvLayer_hpp
#define ConvLayer_hpp

#include "ConvLayerPtr.h"
#include "Layer.h"

namespace astra {

    class ConvLayer : public Layer {
    public:
        static LayerPtr createConvLayerPtr(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long filterWidth, unsigned long filterHeight, unsigned long nFilters);

    public:
        ConvLayer(unsigned long width, unsigned long height, unsigned long nChannels, unsigned long filterWidth, unsigned long filterHeight, unsigned long nFilters);

    public:
        virtual LayerTrainerPtr createTrainer();
        virtual const math::Vector& process(const math::Vector& input);

    public:
        unsigned long getWidth() const {
            return width;
        }

        void setWidth(unsigned long width) {
            ConvLayer::width = width;
        }

        unsigned long getHeight() const {
            return height;
        }

        unsigned long getNChannels() const {
            return nChannels;
        }

        unsigned long getFilterWidth() const {
            return filterWidth;
        }

        unsigned long getFilterHeight() const {
            return filterHeight;
        }

        unsigned long getNFilters() const {
            return nFilters;
        }

        const math::Matrix &getInputCols() const {
            return inputCols;
        }

        const math::Matrix &getLinearMultiplicationResult() const {
            return linearMultiplicationResult;
        }

    protected:
        void setInputCols(const math::Matrix &inputCols) {
            ConvLayer::inputCols = inputCols;
        }

        void setLinearMultiplicationResult(const math::Matrix &linearMultiplicationResult) {
            ConvLayer::linearMultiplicationResult = linearMultiplicationResult;
        }

    private:
        unsigned long width, height;
        unsigned long nChannels;
        unsigned long filterWidth, filterHeight;
        unsigned long nFilters;

        math::Matrix inputCols;
        math::Matrix linearMultiplicationResult;
    };

};

#endif // ConvLayer_hpp