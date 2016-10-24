//
// Created by Pavel on 19/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "Image2Cols.h"
#include <cassert>

using namespace astra::math;

namespace astra {
namespace algorithms {

    MatrixPtr Image2Cols::convertToCols(const std::vector<MatrixPtr>& src, unsigned long kernelWidth, unsigned long kernelHeight, unsigned long stride/* = 1*/, unsigned long padWidth/* = 0*/, unsigned long padHeight/* = 0*/) {
        assert(src.size() > 0);

        unsigned long srcWidth = src[0]->get_width();
        unsigned long srcHeight = src[0]->get_height();

        bool hasPadding = padWidth > 0 || padHeight > 0;
        const std::vector<MatrixPtr>& padSrc = hasPadding ? addPadding(src, padWidth, padHeight) : src;

        unsigned long kernelsCountH = kernelsCount(srcWidth, kernelWidth, stride, padWidth);
        unsigned long kernelsCountV = kernelsCount(srcHeight, kernelHeight, stride, padHeight);

        unsigned long resultWidth = kernelsCountH * kernelsCountV;
        unsigned long resultHeight = kernelWidth * kernelHeight * src.size();

        Matrix result(resultWidth, resultHeight);

        unsigned long x = 0;
        unsigned long y = 0;
        unsigned long kernelCol = 0;

        for (unsigned long row = 0; row < kernelsCountV; ++row, y += stride) {
            for (unsigned long col = 0, x = 0; col < kernelsCountH; ++col, x += stride, ++kernelCol) {
                for (unsigned long i = 0; i < src.size(); ++i) {
                    MatrixPtr sm = padSrc[i]->submatrix(x, y, kernelWidth, kernelHeight);
                    MatrixPtr dm = result.submatrix(kernelCol, 0, 1, resultHeight);
                    std::copy(sm->begin(), sm->end(), dm->begin());
                }
            }
        }

        return std::make_shared<Matrix>(result);
    }

    unsigned long Image2Cols::kernelsCount(unsigned long size, unsigned long kernelSize, unsigned long stride/* = 1*/, unsigned long padSize/* = 0*/) {
        return (size + 2*padSize - kernelSize) / stride + 1;
    }

    std::vector<MatrixPtr> Image2Cols::addPadding(const std::vector<MatrixPtr>& src, unsigned long padWidth, unsigned long padHeight) {
        std::vector<MatrixPtr> result;
        std::for_each(src.begin(), src.end(), [&result, padWidth, padHeight](const MatrixPtr& mat) {
            result.push_back(std::make_shared<Matrix>(Matrix::copy(*mat, padWidth, padHeight)));
        });
        return result;
    }

}}