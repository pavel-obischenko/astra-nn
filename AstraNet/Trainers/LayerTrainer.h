//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_LAYERTRAINER_H
#define ASTRA_NN_LAYERTRAINER_H


#include "LayerTrainerPtr.h"

#include "../Layers/Layer.h"

#include "../Math/Vector.h"
#include "../Math/Matrix.h"

namespace astra {

    class LayerTrainer {
    public:
        explicit LayerTrainer(const LayerPtr& layerPtr) : layerPtr(layerPtr), localGradient(1), errorFactor(1), prevLayerErrorFactor(1), newWeights(1, 1) {}
        virtual ~LayerTrainer() {}

    public:
        virtual math::Vector trainingError(const math::Vector& out, const math::Vector& train) = 0;
        virtual const math::Vector& backpropagateError(const math::Vector& errorFactor, double epsilon) = 0;

    public:
        const LayerPtr& getLayerPtr() const { return layerPtr; }

        const math::Vector& getLocalGradient() const { return localGradient; }
        const math::Vector& getErrorFactor() const { return errorFactor; }
        const math::Vector& getPrevLayerErrorFactor() const { return prevLayerErrorFactor; }
        const math::Matrix& getNewWeights() const { return newWeights; }

    protected:
        void setLocalGradient(const math::Vector& gradient) { localGradient = gradient; }
        void setErrorFactor(const math::Vector& factor) { errorFactor = factor; }
        void setPrevLayerErrorFactor(const math::Vector& factor) { prevLayerErrorFactor = factor; }
        void setNewWeights(const math::Matrix& weights) { newWeights = weights; }

    private:
        LayerPtr layerPtr;

        math::Vector localGradient;
        math::Vector errorFactor;
        math::Vector prevLayerErrorFactor;
        math::Matrix newWeights;
    };
}


#endif //ASTRA_NN_LAYERTRAINER_H
