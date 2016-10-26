//
// Created by Pavel on 25/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_LAYERTRAINERPTR_H
#define ASTRA_NN_LAYERTRAINERPTR_H

#include <memory>
#include <vector>

namespace astra {

    class LayerTrainer;

    typedef std::shared_ptr<LayerTrainer> LayerTrainerPtr;
    typedef std::vector<LayerTrainerPtr> LayerTrainerArray;
    typedef std::shared_ptr<LayerTrainerArray> LayerTrainerArrayPtr;

}


#endif //ASTRA_NN_LAYERTRAINERPTR_H
