//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_LAYERTRAINER_H
#define ASTRA_NN_LAYERTRAINER_H

#include "LayerTrainerPtr.h"

#include "../Layers/Layer.h"
#include "../Math/Math.h"

namespace astra {

    class LayerTrainer {
    public:
        explicit LayerTrainer(const LayerPtr& layerPtr) : layerPtr(layerPtr), localGradient(layerPtr->getNOutputs()), errorFactor(layerPtr->getNInputs()), prevLayerErrorFactor(layerPtr->getNOutputs()), newWeights(layerPtr->getWeights()), dWeights(layerPtr->getWeights()) {
            localGradient.zeroFill();
            errorFactor.zeroFill();
            prevLayerErrorFactor.zeroFill();
            dWeights.zeroFill();
        }
        virtual ~LayerTrainer() {}

    public:
        virtual math::Vector trainingError(const math::Vector& out, const math::Vector& train) {
            return train - out;
        }
        virtual const math::Vector& backpropagateError(const math::Vector& errorFactor, double epsilon, double momentum) = 0;

    protected:
        void adjustWeights(const math::Matrix& dWeights, double momentum) {
            auto lastDWeights = getDWeights();
            // new weights
            setNewWeights(getNewWeights() + dWeights + momentum * lastDWeights);
        }

    public:
        const LayerPtr& getLayerPtr() const { return layerPtr; }

        const math::Vector& getLocalGradient() const { return localGradient; }
        const math::Vector& getErrorFactor() const { return errorFactor; }
        const math::Vector& getPrevLayerErrorFactor() const { return prevLayerErrorFactor; }
        const math::Matrix& getNewWeights() const { return newWeights; }
        const math::Matrix &getDWeights() const { return dWeights; }

    protected:
        void setLocalGradient(const math::Vector& gradient) { localGradient = gradient; }
        void setErrorFactor(const math::Vector& factor) { errorFactor = factor; }
        void setPrevLayerErrorFactor(const math::Vector& factor) { prevLayerErrorFactor = factor; }
        void setNewWeights(const math::Matrix& weights) { newWeights = weights; }
        void setDWeights(const math::Matrix &dWeights) { LayerTrainer::dWeights = dWeights; }

    private:
        LayerPtr layerPtr;

        math::Vector localGradient;
        math::Vector errorFactor;
        math::Vector prevLayerErrorFactor;
        math::Matrix newWeights;
        math::Matrix dWeights;
    };
}

#endif //ASTRA_NN_LAYERTRAINER_H