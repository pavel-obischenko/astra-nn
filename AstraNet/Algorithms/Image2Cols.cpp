//
// Created by Pavel on 19/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "Image2Cols.h"

using namespace astra::math;

namespace astra {
namespace algorithms {

    std::vector<MatrixPtr> addPadding(const std::vector<MatrixPtr>& src, unsigned long padWidth, unsigned long padHeight);

    Matrix image2Cols(const std::vector<MatrixPtr>& src, unsigned long kernelWidth, unsigned long kernelHeight, unsigned long padWidth, unsigned long padHeight, unsigned long stride) {
        bool hasPadding = padWidth > 0 || padHeight > 0;
        const std::vector<MatrixPtr>& psrc = hasPadding ? addPadding(src, padWidth, padHeight) : src;

        return Matrix(1, 1);
    }

    std::vector<MatrixPtr> addPadding(const std::vector<MatrixPtr>& src, unsigned long padWidth, unsigned long padHeight) {
        std::vector<MatrixPtr> result;
        std::for_each(src.begin(), src.end(), [&result, padWidth, padHeight](const MatrixPtr& mat) {
            result.push_back(std::make_shared<Matrix>(Matrix::copy(*mat, padWidth, padHeight)));
        });
        return result;
    }

}}