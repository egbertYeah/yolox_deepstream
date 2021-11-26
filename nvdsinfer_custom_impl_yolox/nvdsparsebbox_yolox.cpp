/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
// #include <omp.h>
// #include <opencv2/opencv.hpp>
#include "nvdsinfer_custom_impl.h"

#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3
static const int NUM_CLASSES = 1;

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int IMAGE_W = 2048;
static const int IMAGE_H = 3072;

const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output";

static constexpr int LOCATIONS = 4;
struct alignas(float) Detection{
    //center_x center_y w h
    float bbox[LOCATIONS];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static void generate_grids_and_stride(std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (auto stride : strides)
    {
        int num_grid_y = INPUT_H / stride;
        int num_grid_x = INPUT_W / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++)
        {
            for (int g0 = 0; g0 < num_grid_x; g0++)
            {
                /* 用于预测每层特征图上anchor的位置信息*/
                grid_strides.push_back((GridAndStride){g0, g1, stride});
            }
        }
    }
}

static void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, float* feat_blob, float prob_threshold, std::vector<Detection>& objects)
{

    const int num_anchors = grid_strides.size(); // 8400

    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

        // yolox/models/yolo_head.py decode logic
        // decode之后的bbox信息
        float x_center = (feat_blob[basic_pos+0] + grid0) * stride;
        float y_center = (feat_blob[basic_pos+1] + grid1) * stride;
        float w = exp(feat_blob[basic_pos+2]) * stride;
        float h = exp(feat_blob[basic_pos+3]) * stride;

        float box_objectness = feat_blob[basic_pos+4];
        for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Detection obj;
                obj.bbox[0] = x_center;
                obj.bbox[1] = y_center;
                obj.bbox[2] = w;
                obj.bbox[3] = h;
                obj.class_id = class_idx;
                obj.conf = box_prob;

                objects.push_back(obj);
            }

        } // class loop

    } // point anchor loop
}

bool cmp(Detection& a, Detection& b) 
{
    return a.conf > b.conf;
}

float iou(float lbox[4], float rbox[4]) 
{
    float interBox[] = 
    {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

void nms_bboxes(std::vector<Detection>& proposals, std::vector<Detection>& res,float nms_thresh) 
{
    // int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (unsigned int i = 0; i < proposals.size(); i++) 
    {
        Detection det = proposals[i];
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) 
    {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) 
        {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) 
            {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) 
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

static void decode_outputs(float* prob, std::vector<Detection>& objects, float scale, const int img_w, const int img_h) {
        std::vector<Detection> proposals;
        std::vector<int> strides = {8, 16, 32};
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(strides, grid_strides);
        generate_yolox_proposals(grid_strides, prob,  BBOX_CONF_THRESH, proposals);
        std::cout << "num of boxes before nms: " << proposals.size() << std::endl;
        // NMS 
        nms_bboxes(proposals, objects, NMS_THRESH);
        // objects 中保存着NMS之后object的信息

        std::cout << "num of boxes: " << objects.size() << std::endl;
}


/* This is a sample bounding box parsing function for the sample YoloV5 detector model */
static bool NvDsInferParseYolox(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    float* prob = (float*)outputLayersInfo[0].buffer;
    std::vector<Detection> objects;

    float scale = std::min(INPUT_W / (IMAGE_W*1.0), INPUT_H / (IMAGE_H*1.0));
    decode_outputs(prob, objects, scale, IMAGE_W, IMAGE_H);
    
    for(auto& r : objects) {
	    NvDsInferParseObjectInfo oinfo;
        
	    oinfo.classId = r.class_id;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]-r.bbox[2]*0.5f);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]-r.bbox[3]*0.5f);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]);
	    oinfo.detectionConfidence = r.conf;
	    objectList.push_back(oinfo);     
    }
    return true;
}

extern "C" bool NvDsInferParseCustomYolox(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseYolox(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolox);
