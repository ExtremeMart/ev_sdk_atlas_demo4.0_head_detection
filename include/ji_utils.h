/*
 * Copyright (c) 2021 Extreme Vision Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef JI_UTILS
#define JI_UTILS

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/freetype.hpp>
#include <fstream>
#include <glog/logging.h>
#include "WKTParser.h"

#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)
#define YUV420SP_SIZE(width, height) ((width) * (height) * 3 / 2)

#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)

#define DEFAULT_FONT_PATH "/usr/local/ev_sdk/lib/fonts/NotoSansCJKsc-Regular.otf"

/**
 * 获取文件大小
 *
 * @param ifs 打开的文件
 * @return 文件大小
 */
static size_t getFileLen(std::ifstream &ifs) {
    int origPos = ifs.tellg();
    ifs.seekg(0, std::fstream::end);
    size_t len = ifs.tellg();
    ifs.seekg(origPos);
    return len;
}

/**
 * 在图上画矩形框，并在框顶部画文字
 *
 * @param img   需要画的图
 * @param leftTopRightBottomRect    矩形框(x, y, width, height)，其中(x, y)是左上角坐标，(width, height)是框的宽高
 * @param text  需要画的文字
 * @param rectLineThickness 矩形框的线宽度
 * @param rectLineType 矩形框的线类型，当值小于0时，将使用颜色填充整个矩形框
 * @param rectColor    矩形框的颜色
 * @param alpha     矩形框的透明度，范围[0,1]
 * @param fontHeight    字体高度
 * @param textColor 字体颜色，BGR格式
 * @param textBg    字体背景颜色，BGR格式
 */
static void drawRectAndText(cv::Mat &img, cv::Rect &leftTopRightBottomRect, const std::string &text, int rectLineThickness,
                            int rectLineType, cv::Scalar rectColor, float rectAlpha, int fontHeight, cv::Scalar textColor,
                            cv::Scalar textBg) {
    cv::Mat originalData;
    if (rectAlpha < 1.0f && rectAlpha > 0.0f) {
        img.copyTo(originalData);
    }
    // Draw rectangle
    cv::Point rectLeftTop(leftTopRightBottomRect.x, leftTopRightBottomRect.y);
    cv::rectangle(img, leftTopRightBottomRect, rectColor, rectLineThickness, rectLineType, 0);

    // Draw text and text background
    cv::Ptr<cv::freetype::FreeType2> ft2;
    int baseline = 0;
    ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData(DEFAULT_FONT_PATH, 0);

    cv::Size textSize = ft2->getTextSize(text, fontHeight, -1, &baseline);
    cv::Point textLeftBottom(leftTopRightBottomRect.x, leftTopRightBottomRect.y);
    textLeftBottom -= cv::Point(0, rectLineThickness);
    textLeftBottom -= cv::Point(0, baseline);   // (left, bottom) of text
    cv::Point textLeftTop(textLeftBottom.x, textLeftBottom.y - textSize.height);    // (left, top) of text
    // Draw text background
    cv::rectangle(img, textLeftTop, textLeftTop + cv::Point(textSize.width, textSize.height + baseline), textBg,
                  cv::FILLED);
    // Draw text
    ft2->putText(img, text, textLeftBottom, fontHeight, textColor, -1, cv::LINE_AA, true);

    if (!originalData.empty()) {    // Need to transparent drawing with alpha
        cv::addWeighted(originalData, rectAlpha, img, (1 - rectAlpha), 0, img);
    }
}

/**
 * 在输入图img上画多边形框
 *
 * @param img   输入图
 * @param polygons  多边形数组，每个多边形由顺时针连接的点构成
 * @param color     多边形框的颜色，BGR格式
 * @param alpha     多边形框的透明度，范围[0,1]
 * @param lineType  多边形框的线类型
 * @param thickness 多边形框的宽度
 * @param isFill    是否使用颜色填充roi区域
 */
static void drawPolygon(cv::Mat &img, std::vector<std::vector<cv::Point> > polygons, const cv::Scalar &color, float alpha,
                        int lineType, int thickness, bool isFill) {
    cv::Mat originalData;
    bool fill = (isFill && alpha < 1.0f && alpha > 0.0f);
    if (fill) {
        img.copyTo(originalData);
    }
    for (size_t i = 0; i < polygons.size(); i++) {
        const cv::Point *pPoint = &polygons[i][0];
        int n = (int) polygons[i].size();
        if (fill) {
            cv::fillPoly(img, &pPoint, &n, 1, color, lineType);
        } else {
            cv::polylines(img, &pPoint, &n, 1, true, color, thickness, lineType);
        }
    }
    if (!originalData.empty()) { // Transparent drawing
        cv::addWeighted(originalData, alpha, img, (1 - alpha), 0, img);
    }
}

/**
 * 在img上画文字text
 *
 * @param img   输入图
 * @param text  文字
 * @param fontHeight    文字大小
 * @param fgColor   文字颜色，BGR格式
 * @param bgColor   文字背景颜色，BGR格式
 * @param leftTop   所画文字的左上顶点所在位置
 */
static void drawText(cv::Mat &img, const std::string &text, int fontHeight, const cv::Scalar &fgColor,
                     const cv::Scalar &bgColor, const cv::Point &leftTopShift) {
    if (text.empty()) {
        printf("text cannot be empty!\n");
        return;
    }

    cv::Ptr<cv::freetype::FreeType2> ft2;
    int baseline = 0;
    ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData(DEFAULT_FONT_PATH, 0);
    cv::Size textSize = ft2->getTextSize(text, fontHeight, -1, &baseline);
    cv::Point textLeftBottom(0, textSize.height);
    textLeftBottom -= cv::Point(0, baseline);   // (left, bottom) of text
    cv::Point textLeftTop(textLeftBottom.x, textLeftBottom.y - textSize.height);    // (left, top) of text
    // Draw text background
    textLeftTop += leftTopShift;
    cv::rectangle(img, textLeftTop, textLeftTop + cv::Point(textSize.width, textSize.height + baseline), bgColor,
                  cv::FILLED);
    textLeftBottom += leftTopShift;
    ft2->putText(img, text, textLeftBottom, fontHeight, fgColor, -1, cv::LINE_AA, true);
}

static float calcMIOU(const cv::Rect &bbox1, const cv::Rect &bbox2) {
    return static_cast<float>((bbox1 & bbox2).area()) / static_cast<float>(std::min(bbox1.area(), bbox2.area()));
}

static float calcIOU(const cv::Rect &bbox1, const cv::Rect &bbox2) {
    return static_cast<float>((bbox1 & bbox2).area()) / static_cast<float>((bbox1 | bbox2).area());
}

static bool bottomInRois(const cv::Rect &box, const VectorPolygon &rois) {
    cv::Point2f bottomCenter(static_cast<float>(box.x) + static_cast<float>(box.width) / 2.0F, static_cast<float>(box.y) + static_cast<float>(box.height));
    for (auto &roi: rois) {
        if (cv::pointPolygonTest(roi, bottomCenter, false) >= 0.0) {
            return true;
        }
    }
    return false;
}

static bool centerInRois(const cv::Rect &box, const VectorPolygon &rois) {
    cv::Point2f center(static_cast<float>(box.x) + static_cast<float>(box.width) / 2.0F, static_cast<float>(box.y) + static_cast<float>(box.height) / 2.0F);
    for (auto &roi: rois) {
        if (cv::pointPolygonTest(roi, center, false) >= 0.0) {
            return true;
        }
    }
    return false;
}

template<typename VECT>
static void maintainVectorSize(std::vector<VECT> &vec, int size) {
    if (vec.size() > size) {
        int erase_length = vec.size() - size;
        vec.erase(vec.begin(), vec.begin() + erase_length);
    }
}

static void Mat_BGR2YUV_nv12(cv::Mat &src, cv::Mat &dst)
{
    auto swapYUV_I420toNV12 = [](unsigned char* i420bytes, unsigned char* nv12bytes, int width, int height)
    {
        int nLenY = width * height;
        int nLenU = nLenY / 4;

        memcpy(nv12bytes, i420bytes, width * height);

        for (int i = 0; i < nLenU; i++)
        {
            nv12bytes[nLenY + 2 * i] = i420bytes[nLenY + i];                    // U
            nv12bytes[nLenY + 2 * i + 1] = i420bytes[nLenY + nLenU + i];        // V
        }
    };

    int w_img = src.cols;
    int h_img = src.rows;
    // align up
    int align_width = ALIGN_UP16(src.cols);
    int align_height = ALIGN_UP2(src.rows);

    if (align_width == w_img && align_height == h_img) {
        dst = cv::Mat(h_img * 1.5, w_img, CV_8UC1, cv::Scalar(0));
        cv::Mat src_YUV_I420(h_img * 1.5, w_img, CV_8UC1, cv::Scalar(0)); //YUV_I420
        cvtColor(src, src_YUV_I420, cv::COLOR_BGR2YUV_I420);
        swapYUV_I420toNV12(src_YUV_I420.data, dst.data, w_img, h_img);
    }
    else {
        cv::Mat align_src(align_height, align_width, CV_8UC3);
        src.copyTo(align_src(cv::Rect(0, 0, w_img, h_img)));
        dst = cv::Mat(align_height * 1.5, align_width, CV_8UC1, cv::Scalar(0));
        cv::Mat src_YUV_I420(align_height * 1.5, align_width, CV_8UC1, cv::Scalar(0)); //YUV_I420
        cvtColor(align_src, src_YUV_I420, cv::COLOR_BGR2YUV_I420);
        swapYUV_I420toNV12(src_YUV_I420.data, dst.data, align_width, align_height);
    }
}


#define SDKLOG(b) LOG(b)<<" [SDKLOG] "
#define SDKLOG_FIRST_N(b,i) LOG_FIRST_N(b,i)<<" [SDKLOG] "

#endif  // JI_UTILS
