//
// Created by Pavel on 19/10/16.
// Copyright (c) 2016 pavel. All rights reserved.
//

#include "Image2Cols.h"
#include <cassert>

using namespace astra::math;

namespace astra {
namespace algorithms {

    MatrixPtr Image2Cols::convertToCols(const math::Vector& src, unsigned long nChannels, unsigned long kernelWidth, unsigned long kernelHeight, bool addBiasInput/* = false*/, unsigned long stride/* = 1*/, unsigned long padWidth/* = 0*/, unsigned long padHeight/* = 0*/) {
        std::vector<MatrixPtr> srcVec;
        unsigned long channelSize = src.size() / nChannels;
        unsigned long width = (unsigned long)std::sqrt(channelSize);
        unsigned long height = (unsigned long)std::sqrt(channelSize);

        for (unsigned long i = 0; i < nChannels; ++i) {
            unsigned long index = i * channelSize;
            MatrixPtr m = std::make_shared<Matrix>(Matrix(width, height));

            auto begin = src.get_data_storage()->begin() + index;
            auto end = begin + channelSize;
            std::copy(begin, end, m->begin());

            srcVec.push_back(m);
        }

        return convertToCols(srcVec, kernelWidth, kernelHeight, addBiasInput, stride, padWidth, padHeight);
    }

    MatrixPtr Image2Cols::convertToCols(const std::vector<MatrixPtr>& src, unsigned long kernelWidth, unsigned long kernelHeight, bool addBiasInput/* = false*/, unsigned long stride/* = 1*/, unsigned long padWidth/* = 0*/, unsigned long padHeight/* = 0*/) {
        assert(src.size() > 0);

        unsigned long srcWidth = src[0]->get_width();
        unsigned long srcHeight = src[0]->get_height();

        bool hasPadding = padWidth > 0 || padHeight > 0;
        const std::vector<MatrixPtr>& padSrc = hasPadding ? addPadding(src, padWidth, padHeight) : src;

        unsigned long kernelsCountH = kernelsCount(srcWidth, kernelWidth, stride, padWidth);
        unsigned long kernelsCountV = kernelsCount(srcHeight, kernelHeight, stride, padHeight);

        unsigned long resultWidth = kernelsCountH * kernelsCountV;
        unsigned long resultHeight = kernelWidth * kernelHeight * src.size() + (addBiasInput ? 1 : 0);

        Matrix result(resultWidth, resultHeight);

        unsigned long y = 0;
        unsigned long kernelCol = 0;

        for (unsigned long row = 0; row < kernelsCountV; ++row, y += stride) {
            for (unsigned long col = 0, x = 0; col < kernelsCountH; ++col, x += stride, ++kernelCol) {
                MatrixPtr dm = result.submatrix(kernelCol, 0, 1, resultHeight);
                auto begin = dm->begin();

                for (unsigned long i = 0; i < src.size(); ++i) {
                    MatrixPtr sm = padSrc[i]->submatrix(x, y, kernelWidth, kernelHeight);
                    begin = std::copy(sm->begin(), sm->end(), begin);
                }

                if (addBiasInput) {
                    auto end = dm->end();
                    *(--end) = 1;
                }
            }
        }

        return std::make_shared<Matrix>(result);
    }

    std::vector<math::MatrixPtr> Image2Cols::convertFromColsToArray(const math::Matrix& srcCols, unsigned long width, unsigned long height, unsigned long nChannels, unsigned long kernelWidth, unsigned long kernelHeight, bool cutBiases/* = false*/, unsigned long stride/* = 1*/) {
        std::vector<MatrixPtr> resVec;

        for (unsigned long ch = 0; ch < nChannels; ++ch) {
            // TODO: optimize
            resVec.push_back(std::make_shared<Matrix>(Matrix::zero(width, height)));
        }

        unsigned long y = 0;
        unsigned long kernelCol = 0;
        unsigned long kernelSize = kernelWidth * kernelHeight;

        unsigned long kernelsCountH = kernelsCount(width, kernelWidth, stride, 0);
        unsigned long kernelsCountV = kernelsCount(height, kernelHeight, stride, 0);

        for (unsigned long row = 0; row < kernelsCountV; ++row, y += stride) {
            for (unsigned long col = 0, x = 0; col < kernelsCountH; ++col, x += stride, ++kernelCol) {
                for (unsigned long ch = 0; ch < nChannels; ++ch) {
                    unsigned long h = (srcCols.get_height() - (cutBiases ? 1 : 0)) / nChannels;
                    ConstMatrixPtr sm = srcCols.submatrix(kernelCol, ch * kernelSize, 1, h);
                    MatrixPtr dm = resVec[ch]->submatrix(x, y, kernelWidth, kernelHeight);
                    std::transform(sm->begin(), sm->end(), dm->begin(), dm->begin(), std::plus<double>());
                }
            }
        }
        return resVec;
    }

    math::VectorPtr Image2Cols::convertFromColsToVector(const math::Matrix& srcCols, unsigned long width, unsigned long height, unsigned long nChannels, unsigned long kernelWidth, unsigned long kernelHeight, bool cutBiases/* = false*/, unsigned long stride/* = 1*/) {
        VectorPtr result = std::make_shared<Vector>(width * height * nChannels);
        auto begin = result->begin();

        auto resVec = convertFromColsToArray(srcCols, width, height, nChannels, kernelWidth, kernelHeight, cutBiases, stride);
        for (unsigned long ch = 0; ch < nChannels; ++ch) {
            begin = std::copy(resVec[ch]->begin(), resVec[ch]->end(), begin);
        }
        return result;
    }


    unsigned long Image2Cols::kernelsCount(unsigned long size, unsigned long kernelSize, unsigned long stride/* = 1*/, unsigned long padSize/* = 0*/) {
        return 1 + (size + 2*padSize - kernelSize) / stride;
    }

    std::vector<MatrixPtr> Image2Cols::addPadding(const std::vector<MatrixPtr>& src, unsigned long padWidth, unsigned long padHeight) {
        std::vector<MatrixPtr> result;
        std::for_each(src.begin(), src.end(), [&result, padWidth, padHeight](const MatrixPtr& mat) {
            result.push_back(std::make_shared<Matrix>(Matrix::copy(*mat, padWidth, padHeight)));
        });
        return result;
    }

}}