//
// Created by Pavel on 19/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#ifndef ASTRA_NN_IMAGE2COLS_H
#define ASTRA_NN_IMAGE2COLS_H

#include "../Math/Math.h"

namespace astra {
namespace algorithms {

    class Image2Cols {
    public:
        static math::MatrixPtr convertToCols(const math::Vector& src, unsigned long nChannels, unsigned long kernelWidth, unsigned long kernelHeight, bool addBiasInput = false, unsigned long stride = 1, unsigned long padWidth = 0, unsigned long padHeight = 0);
        static math::MatrixPtr convertToCols(const std::vector<math::MatrixPtr> &src, unsigned long kernelWidth, unsigned long kernelHeight, bool addBiasInput = false, unsigned long stride = 1, unsigned long padWidth = 0, unsigned long padHeight = 0);

        static std::vector<math::MatrixPtr> convertFromColsToArray(const math::Matrix& srcCols, unsigned long width, unsigned long height, unsigned long nChannels, unsigned long kernelWidth, unsigned long kernelHeight, bool cutBiases = false, unsigned long stride = 1);
        static math::VectorPtr convertFromColsToVector(const math::Matrix& srcCols, unsigned long width, unsigned long height, unsigned long nChannels, unsigned long kernelWidth, unsigned long kernelHeight, bool cutBiases = false, unsigned long stride = 1);

        static unsigned long kernelsCount(unsigned long size, unsigned long kernelSize, unsigned long stride = 1, unsigned long padSize = 0);

    private:
        static std::vector<math::MatrixPtr> addPadding(const std::vector<math::MatrixPtr>& src, unsigned long padWidth, unsigned long padHeight);
    };
}}

#endif //ASTRA_NN_IMAGE2COLS_H
