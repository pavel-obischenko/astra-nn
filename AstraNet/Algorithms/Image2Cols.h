//
// Created by Pavel on 19/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_IMAGE2COLS_H
#define ASTRA_NN_IMAGE2COLS_H

#include "../Math/Matrix.h"

namespace astra {
namespace algorithms {

    math::Matrix image2Cols(const std::vector<math::MatrixPtr>& src, unsigned long kernelWidth, unsigned long kernelHeight, unsigned long padWidth, unsigned long padHeight, unsigned long stride);
    std::vector<math::MatrixPtr> addPadding(const std::vector<math::MatrixPtr>& src, unsigned long padWidth, unsigned long padHeight);

}}

#endif //ASTRA_NN_IMAGE2COLS_H
