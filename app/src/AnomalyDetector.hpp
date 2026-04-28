#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <faiss/IndexFlat.h>
#include <atomic>
#include <string>
#include <vector>

struct DetectionResult {
    bool        is_defect = false;
    float       anomaly_score = 0.0f;
    float       threshold = 0.50f;
    cv::Mat     heatmap;
    cv::Point   defect_location{ 0, 0 };
    float       defect_area = 0.0f;
    float       defect_area_mm2 = 0.0f;
    float       defect_size_mm = 0.0f;
    int         frame_id = 0;
    std::string image_path;
    std::string timestamp;
};

class AnomalyDetector {
public:
    AnomalyDetector(const std::string& onnx_path,
        const std::string& memory_bank_path,
        float threshold = 0.50f);
    ~AnomalyDetector();
    bool isReady() const;
    DetectionResult inspect(const cv::Mat& image,
        const std::string& image_path = "");
    void resetFrameCounter() { frame_counter_ = 0; }
    void setFrameCounter(int n) { frame_counter_ = n; }
    int  getFrameCounter() { return frame_counter_.load(); }
private:
    Ort::Env            env_;
    Ort::Session* session_ = nullptr;
    Ort::SessionOptions session_options_;
    cv::Mat             memory_bank_;
    faiss::IndexFlatL2* faiss_index_ = nullptr;
    float               threshold_;
    bool                ready_ = false;
    std::atomic<int>    frame_counter_{ 0 };

    cv::Mat     preprocess(const cv::Mat& image);
    cv::Mat     extractFeatures(const cv::Mat& blob);
    cv::Mat     computeAnomalyMap(const cv::Mat& features);
    std::string currentTimestamp();
};