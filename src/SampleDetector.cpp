#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
//jsoncpp 相关的头文件
#include "reader.h"
#include "writer.h"
#include "value.h"
#include "SampleDetector.hpp"

SampleDetector::SampleDetector()
{    
}

SampleDetector::~SampleDetector()
{
    UnInit();
}

STATUS SampleDetector::Init(const std::string &strModelName, float thresh)
{        
   //如果已经初始化,则直接返回
    if (mInitialized)
    {
        SDKLOG(INFO) << "AlgoJugement instance is initied already!";
        return STATUS_SUCCESS;
    }

    // 初始化ACL资源
    STATUS status = InitAcl();
    if(status != STATUS_SUCCESS)
    {
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "succeed to init acl";
    //初始化模型管理实例
    status = InitModel(strModelName);
    if(status != STATUS_SUCCESS)
    {
        return ERROR_INITMODEL;
    }
    SDKLOG(INFO) << "succeed to init model";
    status = InitDvpp();
    if(status != STATUS_SUCCESS)
    {
        return ERROR_INITDVPP;
    }
    SDKLOG(INFO) << "succeed to init dvpp";
    mInitialized = true;
    return STATUS_SUCCESS;
}

STATUS SampleDetector::InitAcl()
{
    mAclContext = nullptr;
    mAclStream = nullptr;    
    uint32_t deviceCount;
    aclError ret = aclrtGetDeviceCount(&deviceCount);
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "No device found! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << deviceCount << " devices found";

    // open device
    ret = aclrtSetDevice(mDeviceId);
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "Acl open device " << mDeviceId << " failed! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "Open device " << mDeviceId << " success";

    // create context (set current)
    ret = aclrtCreateContext(&mAclContext, mDeviceId);
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "acl create context failed! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "create context success";

    // set current context
    ret = aclrtSetCurrentContext(mAclContext);
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "acl set context failed! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "set context success";

    // create stream
    ret = aclrtCreateStream(&mAclStream);
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "acl create stream failed! aclError= " << ret;
        return ERROR_INITACL;
    }
    SDKLOG(INFO) << "create stream success";

    //获取当前应用程序运行在host还是device
    ret = aclrtGetRunMode(&mAclRunMode);
    if (ret != ACL_ERROR_NONE)
    {
        SDKLOG(ERROR) << "acl get run mode failed! aclError= " << ret;
        return ERROR_INITACL;
    }
    return STATUS_SUCCESS;
}

STATUS SampleDetector::InitModel(const std::string &strModelName)
{
    SDKLOG(INFO) << "load model " << strModelName;
    auto ret = aclmdlQuerySize(strModelName.c_str(), &mModelMSize, &mModelWSize);
    ret = aclrtMalloc(&mModelMptr, mModelMSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclrtMalloc(&mModelWptr, mModelWSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ret = aclmdlLoadFromFileWithMem(strModelName.c_str(), &mModelID, mModelMptr,mModelMSize, mModelWptr, mModelWSize);
    mModelDescPtr = aclmdlCreateDesc();
    ret = aclmdlGetDesc(mModelDescPtr, mModelID);    
    // 创建模型输出的数据集结构
    mInputDatasetPtr = aclmdlCreateDataset();
    mOutputDatasetPtr = aclmdlCreateDataset();
    //获取模型的输入个数,为输入分配内存并创建输出数据集
    int iModelInputNum = aclmdlGetNumInputs(mModelDescPtr);     
    SDKLOG(INFO) << "num of inputs " << iModelInputNum;
    for (size_t i = 0; i < iModelInputNum; ++i)
    {
        aclmdlIODims dims;
        aclmdlGetInputDims(mModelDescPtr, i, &dims);
        SDKLOG(INFO) << "input dim is : "<< dims.dims[0] << " " << dims.dims[1] << " " << dims.dims[2] << " " << dims.dims[3];
        mModelInputSize = dims.dims[2];
        size_t buffer_size = aclmdlGetInputSizeByIndex(mModelDescPtr, i);
        void *inputBuffer = nullptr;
        //需要和DVPP预处理的数据进行内存复用,所以需要用DVPP的接口分配
        aclError ret = acldvppMalloc(&inputBuffer, buffer_size);
        mVecIBuffers.push_back(inputBuffer);
        aclDataBuffer *inputData = aclCreateDataBuffer(inputBuffer, buffer_size);
        mVecWrappedIOBuffers.push_back(inputData);
        ret = aclmdlAddDatasetBuffer(mInputDatasetPtr, inputData);
    }
    //获取模型的输出个数,为输出分配内存并创建输出数据集
    mModelOutputNums = aclmdlGetNumOutputs(mModelDescPtr);
    SDKLOG(INFO) << "num of outputs " << mModelOutputNums;
    for (size_t i = 0; i < mModelOutputNums; ++i)
    {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(mModelDescPtr, i);
        aclmdlIODims dims;
        aclmdlGetOutputDims(mModelDescPtr, i, &dims);
        SDKLOG(INFO) << "output dims is : " << dims.dims[0] << " " << dims.dims[1] << " " << dims.dims[2] << " " << dims.dims[3];
        mNumClass = dims.dims[1] / 3 - 5;        
        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        mVecOBuffers.push_back(outputBuffer);
        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        mVecWrappedIOBuffers.push_back(outputData);
        ret = aclmdlAddDatasetBuffer(mOutputDatasetPtr, outputData);
    }
    
    aclrtGetRunMode(&mAclRunMode);
    if(mAclRunMode == ACL_DEVICE)
    {
        SDKLOG(INFO) << "run in device mode";
    }
    else
    {
        SDKLOG(INFO) << "run in host mode";
    }
    return STATUS_SUCCESS;
}

STATUS SampleDetector::UnInit()
{    
    if(mInitialized == false)
    {
        return STATUS_SUCCESS;    
    }    
    SDKLOG(INFO) << "in uninit func";
    // 释放输入数据集
    if(mInputDatasetPtr != nullptr)
    {
        aclmdlDestroyDataset(mInputDatasetPtr);
        mInputDatasetPtr = nullptr;
    }
    // 释放输出数据集
    if(mOutputDatasetPtr != nullptr)
    {
        aclmdlDestroyDataset(mOutputDatasetPtr);
        mOutputDatasetPtr = nullptr;
    }
    // 释放模型输入输出的封装结构体
    for( auto& datas :mVecWrappedIOBuffers)
    {
        aclDestroyDataBuffer(datas);
    }
    mVecWrappedIOBuffers.resize(0);
    
    // 释放模型输入内存
    for(auto& ibuffer: mVecIBuffers)
    {        
        acldvppFree(ibuffer);        
    }
    mVecIBuffers.resize(0);

    // 释放模型输出内存
    for(auto& obuffer: mVecOBuffers)
    {        
        aclrtFree(obuffer);        
    }
    mVecOBuffers.resize(0);

    //释放存放模型的相关资源
    if(mModelMptr != nullptr)
    {
        aclrtFree(mModelMptr);
        mModelMptr = nullptr;
    }
    if(mModelWptr != nullptr)
    {
        aclrtFree(mModelWptr);
        mModelWptr = nullptr;
    }
    if(mModelDescPtr != nullptr)
    {
        aclmdlDestroyDesc(mModelDescPtr);
        mModelDescPtr = nullptr;
    }   
    aclmdlUnload(mModelID);

    aclrtDestroyStream(mAclStream);
    aclrtDestroyContext(mAclContext); 
    mInitialized = false;
}

STATUS SampleDetector::ProcessImage(const JiImageInfo &inFrame, std::vector<BoxInfo> &result, float thresh)
{    
    mThresh = thresh;
    mInputHeight = inFrame.nHeight;
    mInputWidth = inFrame.nWidth;
    //利用DVPP运行resize和填充
    STATUS status = PreProcess(inFrame);
    //运行模型推理
    status = doInference();
    //解析yolo层的输出
    std::vector<int> featMapSize{64, 32, 16};
    std::vector<std::pair<int, int>> scaleAnchor1{ {8, 9},  {22, 24},  {39, 54} };
    std::vector<std::pair<int, int>> scaleAnchor2{ {111, 34},  {70, 96}, {125, 135} };
    std::vector<std::pair<int, int>> scaleAnchor3{ {330, 67},  {165, 245}, {341, 296} };
    std::vector<std::vector<std::pair<int, int>>> Anchors = {scaleAnchor1, scaleAnchor2, scaleAnchor3};
    std::vector<BoxInfo> detBoxes{};
    for (size_t i = 0; i < mModelOutputNums; ++i)
    {
        aclDataBuffer *outputBuffer = aclmdlGetDatasetBuffer(mOutputDatasetPtr, i);
        void *data = aclGetDataBufferAddr(outputBuffer);
        size_t bufferSize = aclGetDataBufferSize(outputBuffer);                
        if(mAclRunMode == ACL_HOST)
        {
            void *hostOutputData = new char[bufferSize];        
            aclrtMemcpy(hostOutputData, bufferSize, data, bufferSize, ACL_MEMCPY_DEVICE_TO_HOST);
            parseSingleScale(static_cast<float *>(hostOutputData), featMapSize[i], Anchors[i], detBoxes);
            delete[] hostOutputData;    
        }
        else
        {            
            parseSingleScale(static_cast<float *>(data), featMapSize[i], Anchors[i], detBoxes);
        }        
    }
    runNms(detBoxes);
    result = detBoxes;
    return STATUS_SUCCESS;    
}

STATUS SampleDetector::InitDvpp()
{    
    mDVPPChnDescPtr = acldvppCreateChannelDesc();
    aclError ret = acldvppCreateChannel(mDVPPChnDescPtr); 
    mDvppResizeInPicConfPtr = acldvppCreatePicDesc();   
    mDvppResizeOutPicConfPtr = acldvppCreatePicDesc();       
    mDvppCopyPasteInPicConfptr = acldvppCreatePicDesc();    
    mDvppCopyPasteOutPicConfPtr = acldvppCreatePicDesc();   
    mDvppResizeConfPtr = acldvppCreateResizeConfig();    
    return STATUS_SUCCESS;
}

STATUS SampleDetector::PreProcess(const JiImageInfo &inFrame)
{    
    size_t imDataSize = (inFrame.nFormat == JI_IMAGE_TYPE_YUV420) ? YUV420SP_SIZE(inFrame.nWidthStride, inFrame.nHeightStride) : RGBU8_IMAGE_SIZE(inFrame.nWidthStride, inFrame.nHeightStride);
    auto pixelFormat = (inFrame.nFormat == JI_IMAGE_TYPE_YUV420) ? PIXEL_FORMAT_YUV_SEMIPLANAR_420 : PIXEL_FORMAT_BGR_888;
    void *inputData = nullptr;
    acldvppMalloc(&inputData, imDataSize);
    //设置resize操作的输入参数
    if(mAclRunMode == ACL_HOST)
    {
       aclrtMemcpy(inputData, imDataSize, inFrame.pData, imDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    else
    {        
        aclrtMemcpy(inputData, imDataSize, inFrame.pData, imDataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }  
    acldvppSetPicDescFormat(mDvppResizeInPicConfPtr, pixelFormat);
    acldvppSetPicDescWidth(mDvppResizeInPicConfPtr, inFrame.nWidth);
    acldvppSetPicDescHeight(mDvppResizeInPicConfPtr, inFrame.nHeight);
    acldvppSetPicDescWidthStride(mDvppResizeInPicConfPtr, inFrame.nWidthStride);
    acldvppSetPicDescHeightStride(mDvppResizeInPicConfPtr, inFrame.nHeightStride);
    acldvppSetPicDescSize(mDvppResizeInPicConfPtr, imDataSize);
    acldvppSetPicDescData(mDvppResizeInPicConfPtr, inputData);
    //设置resize操作的输出参数
    mResizeScale = static_cast<float>(mModelInputSize) / std::max(inFrame.nWidth, inFrame.nHeight);
    int resizedWidth = mResizeScale * inFrame.nWidth;
    int resizedHeight = mResizeScale * inFrame.nHeight;    
    imDataSize = YUV420SP_SIZE(ALIGN_UP16(resizedWidth), ALIGN_UP2(resizedHeight));
    void *resizeData = nullptr;
    acldvppMalloc(&resizeData, imDataSize);
    acldvppSetPicDescFormat(mDvppResizeOutPicConfPtr, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(mDvppResizeOutPicConfPtr, resizedWidth);
    acldvppSetPicDescHeight(mDvppResizeOutPicConfPtr, resizedHeight);
    acldvppSetPicDescWidthStride(mDvppResizeOutPicConfPtr, ALIGN_UP16(resizedWidth));
    acldvppSetPicDescHeightStride(mDvppResizeOutPicConfPtr, ALIGN_UP2(resizedHeight));
    acldvppSetPicDescSize(mDvppResizeOutPicConfPtr, imDataSize);
    acldvppSetPicDescData(mDvppResizeOutPicConfPtr, resizeData);
    //设置copy paste操作的输入参数
    acldvppSetPicDescData(mDvppCopyPasteInPicConfptr, resizeData);
    acldvppSetPicDescFormat(mDvppCopyPasteInPicConfptr, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(mDvppCopyPasteInPicConfptr, resizedWidth);
    acldvppSetPicDescHeight(mDvppCopyPasteInPicConfptr, resizedHeight);
    acldvppSetPicDescWidthStride(mDvppCopyPasteInPicConfptr, ALIGN_UP16(resizedWidth));
    acldvppSetPicDescHeightStride(mDvppCopyPasteInPicConfptr, ALIGN_UP2(resizedHeight));
    acldvppSetPicDescSize(mDvppCopyPasteInPicConfptr, YUV420SP_SIZE(ALIGN_UP16(resizedWidth), ALIGN_UP2(resizedHeight)));    
    //设置copy paste 操作的输出参数
    acldvppSetPicDescData(mDvppCopyPasteOutPicConfPtr, mVecIBuffers[0]);
    acldvppSetPicDescFormat(mDvppCopyPasteOutPicConfPtr, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    acldvppSetPicDescWidth(mDvppCopyPasteOutPicConfPtr, mModelInputSize);
    acldvppSetPicDescHeight(mDvppCopyPasteOutPicConfPtr, mModelInputSize);
    acldvppSetPicDescWidthStride(mDvppCopyPasteOutPicConfPtr, mModelInputSize);
    acldvppSetPicDescHeightStride(mDvppCopyPasteOutPicConfPtr, mModelInputSize);
    acldvppSetPicDescSize(mDvppCopyPasteOutPicConfPtr, YUV420SP_SIZE(mModelInputSize, mModelInputSize));
    //设置copy paste 操作的ROI
    mDvppCropConf = acldvppCreateRoiConfig(0, resizedWidth % 2 == 0? resizedWidth - 1 : resizedWidth, 0, resizedHeight % 2 == 0? resizedHeight - 1 : resizedHeight);
    mDvppPasteConf = acldvppCreateRoiConfig(0, resizedWidth % 2 == 0? resizedWidth - 1 : resizedWidth, 0, resizedHeight % 2 == 0? resizedHeight - 1 : resizedHeight);
    
    //等比例缩放后用黑色填充为正方形,这里设置填充为黑色
    aclrtMemset(mVecIBuffers[0], mModelInputSize * mModelInputSize * 1.5, 16, mModelInputSize * mModelInputSize * 1.5);
    aclrtMemset((char*)mVecIBuffers[0] + (mModelInputSize * mModelInputSize), (mModelInputSize * mModelInputSize) / 2, 128, (mModelInputSize * mModelInputSize) / 2);

    //先运行等比例缩放,再运行填充
    aclError ret = acldvppVpcResizeAsync(mDVPPChnDescPtr, mDvppResizeInPicConfPtr, mDvppResizeOutPicConfPtr, mDvppResizeConfPtr, mAclStream);
    ret = aclrtSynchronizeStream(mAclStream);    
    ret = acldvppVpcCropAndPasteAsync(mDVPPChnDescPtr, mDvppCopyPasteInPicConfptr, mDvppCopyPasteOutPicConfPtr, mDvppCropConf, mDvppPasteConf, mAclStream);
    ret = aclrtSynchronizeStream(mAclStream);
    //释放临时数据  
    acldvppFree(inputData);    
    acldvppFree(resizeData);  

    return STATUS_SUCCESS;
}

STATUS SampleDetector::doInference()
{    
    auto ret = aclmdlExecute(mModelID, mInputDatasetPtr, mOutputDatasetPtr);
    return STATUS_SUCCESS;
}

void SampleDetector::parseSingleScale(float *data, size_t featSize, std::vector<std::pair<int, int>> anchors, std::vector<BoxInfo> &vecBoxObjs)
{
    for(int k = 0; k < anchors.size(); ++k)//k
    {
        for (size_t i = 0; i < featSize; ++i)//h
        {
            for (size_t j = 0; j < featSize; ++j)//w
            {
                float x, y, w, h, conf;
                int confPos = j + i * featSize + k * (5 + mNumClass) * (featSize*featSize)  + 4 * featSize * featSize;
                if(data[confPos] > mThresh)
                {
                    x = (j + data[confPos - featSize * featSize * 4]) / featSize * mModelInputSize;
                    y = (i + data[confPos - featSize * featSize * 3]) / featSize * mModelInputSize;
                    w = data[confPos - featSize * featSize * 2];
                    h = data[confPos - featSize * featSize];                    
                    w = w / (1 - w) * anchors[k].first;
                    h = h / (1 - h) * anchors[k].second;
                    int bestId = -1;
                    for (int id = 0; id < mNumClass; ++id)
                    {
                        if (data[confPos] * data[confPos + (id + 1) * featSize * featSize] > mThresh)
                        {
                            bestId = id;
                        }
                    }
                    if(bestId >= 0)
                    {
                        x /= mResizeScale;
                        y /= mResizeScale;
                        w /= mResizeScale;
                        h /= mResizeScale;                        
                        w = x - w / 2 <= 0 ? 2 * x : w;
                        w = x + w / 2 >= mInputWidth  ? 2 * (mInputWidth - x) : w;
                        h = y - h / 2 <= 0 ? 2 * y : h;
                        h = y + h / 2 >= mInputHeight ? 2 * (mInputHeight - y) : h;
                        float score = data[confPos] * data[confPos + (bestId + 1) * featSize * featSize];                                            
                        vecBoxObjs.push_back( BoxInfo{x-w/2, y-h/2, x + w/2, y + h/2, score, bestId});
                    }
                }                
            }
        }
    }
    return;
}

static float BoxIOU(const cv::Rect2d &b1, const cv::Rect2d &b2)
{
    cv::Rect2d inter = b1 & b2;    
    return inter.area() / (b1.area() + b2.area() - inter.area());
}

void SampleDetector::runNms(std::vector<BoxInfo> & vecBoxObjs)
{
    std::sort(vecBoxObjs.begin(), vecBoxObjs.end(), [](const BoxInfo &b1, const BoxInfo &b2){return b1.score > b2.score;});
    for (int i = 0; i < vecBoxObjs.size(); ++i)
    {
        if (vecBoxObjs[i].score == 0)
        {
            continue;
        }
        for (int j = i + 1; j < vecBoxObjs.size(); ++j)
        {
            if (vecBoxObjs[j].score == 0)
            {
                continue;
            }
            cv::Rect pos1{vecBoxObjs[i].x1, vecBoxObjs[i].y1, vecBoxObjs[i].x2-vecBoxObjs[i].x1, vecBoxObjs[i].y2- vecBoxObjs[i].y1};
            cv::Rect pos2{vecBoxObjs[j].x1, vecBoxObjs[j].y1, vecBoxObjs[j].x2-vecBoxObjs[j].x1, vecBoxObjs[j].y2- vecBoxObjs[j].y1};
            if (BoxIOU(pos1, pos2) >= 0.45 )
            {
                vecBoxObjs[j].score = 0;
            }            
        }
    }
    for (auto iter = vecBoxObjs.begin(); iter != vecBoxObjs.end(); ++iter)
    {
        if (iter->score < 0.01)
        {
            vecBoxObjs.erase(iter);
            --iter;
        }
    }
}