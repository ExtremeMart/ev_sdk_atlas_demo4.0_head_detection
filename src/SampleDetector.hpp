/*
 * Copyright (c) 2021 ExtremeVision Co., Ltd.
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
 

#ifndef JI_SAMPLEDETECTOR_HPP
#define JI_SAMPLEDETECTOR_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include "ji.h"
#include "ji_utils.h"
#include "WKTParser.h"
#include "Configuration.hpp"

#define STATUS int

// 表示检测结果的结构体
typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class SampleDetector
{

public:
    /*
     * @breif 检测器构造函数     
    */ 
    SampleDetector();
    
    /*
     * @breif 检测器析构函数     
    */ 
    ~SampleDetector();
    
    /*
     * @breif 初始化检测器相关的资源
     * @param strModelName 检测器加载的模型名称     
     * @param thresh 检测阈值
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    STATUS Init(const std::string &strModelName, float thresh);

    /*
     * @breif 去初始化,释放模型检测器的资源     
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */ 
    STATUS UnInit();
    
    /*
     * @breif 根据送入的图片进行模型推理, 并返回检测结果
     * @param inFrame 输入图片
     * @param result 检测结果通过引用返回
     * @return 返回结果, STATUS_SUCCESS代表调用成功
    */
    STATUS ProcessImage(const JiImageInfo &inFrame, std::vector<BoxInfo> &result, float thresh = 0.15);

private:
    
    /*
     * @breif 初始化CANN运行环境的资源,创建context,stream等对象     
    */
    STATUS InitAcl();
    
    /*
     * @breif 初始化DVPP调用的资源,用于图像预处理,等比例缩放等
    */
    STATUS InitDvpp();
    
    /*
     * @breif 加载推理模型
     * @parma strModelName, om模型的名称
    */
    STATUS InitModel(const std::string &strModelName);
    
    /*
     * @breif 运行图像预处理,采用等比例缩放的方式进行预处理
     * @parma inFrame,输入图像
    */
    STATUS PreProcess(const JiImageInfo &inFrame);
    
    /*
     * @breif 运行模型推理     
    */
    STATUS doInference();

    /*
     * @breif 对yolo层的输出进行解析,恢复出检测框
     * @parma data, 输出yolo层的内存数据指针
     * @parma featSize, 输出yolo层的feature map大小
     * @parma anchors, 输出yolo层对应的anchor大小
     * @parma vecBoxObjs, yolo层输出的检测框
    */
    void parseSingleScale(float *data, size_t featSize, std::vector<std::pair<int, int>> anchors, std::vector<BoxInfo> &vecBoxObjs);
    
    /*
     * @breif 运行nms算法
     * @parma vecBoxObjs, 输出的检测框
    */
    void runNms(std::vector<BoxInfo> & vecBoxObjs);

public:
    // 接口的返回值定义
    static const int ERROR_BASE = 0x0200;
    static const int ERROR_INPUT = 0x0201;
    static const int ERROR_INIT = 0x0202;
    static const int ERROR_PROCESS = 0x0203;
    static const int ERROR_INITACL = 0x0204;
    static const int ERROR_INITMODEL = 0x0205;
    static const int ERROR_INITDVPP = 0x0206;
    static const int STATUS_SUCCESS = 0x0000;   

private:
    aclrtRunMode mAclRunMode;   //运行模式,即当前应用运行在atlas200dk还是AI1    
    aclrtContext mAclContext;
    aclrtStream mAclStream;
    acldvppChannelDesc *mDVPPChnDescPtr{nullptr};
    int32_t mDeviceId = 0;

// 模型推理相关资源
private:
    size_t mModelMSize{0};
    size_t mModelWSize{0};
    void *mModelMptr{nullptr};
    void *mModelWptr{nullptr};
    uint32_t mModelID{0};
    aclmdlDesc *mModelDescPtr{nullptr};
    aclmdlDataset *mInputDatasetPtr{nullptr};
    aclmdlDataset *mOutputDatasetPtr{nullptr};
    size_t mModelInputSize{0};
    int mModelOutputNums{0};
    int mNumClass{0};
    std::vector<void *> mVecIBuffers{};
    std::vector<void *> mVecOBuffers{};
    std::vector<aclDataBuffer *> mVecWrappedIOBuffers{};

// 图像缩放预处理相关资源,主要做等比例缩放操作
private:
    float mResizeScale = 1.f;
    int mInputWidth = 0;
    int mInputHeight = 0;
    int mAlignedInputWidth = 0;
    int mAlignedInputHeight = 0;

    int mResizedWidth = 0;
    int mAlignedResizedWidth = 0;
    int mResizedHeight = 0;
    int mAlignedResizedHeight = 0;

    acldvppResizeConfig *mDvppResizeConfPtr{nullptr};    
    acldvppPicDesc *mDvppResizeOutPicConfPtr{nullptr};
    acldvppPicDesc *mDvppResizeInPicConfPtr{nullptr}; 

    acldvppPicDesc *mDvppCopyPasteOutPicConfPtr{nullptr};
    acldvppPicDesc *mDvppCopyPasteInPicConfptr{nullptr}; // resizeInputDesc_;  // resize input desc

    acldvppRoiConfig *mDvppCropConf{nullptr};
    acldvppRoiConfig *mDvppPasteConf{nullptr};  

private:
    bool mInitialized = false;
    float mThresh = 0.3;

};

#endif //JI_SAMPLEDETECTOR_HPP
