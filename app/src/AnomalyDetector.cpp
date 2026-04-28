#include "AnomalyDetector.hpp"
#include <opencv2/imgproc.hpp>
#include <faiss/IndexFlat.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <limits>
#include <algorithm>

AnomalyDetector::AnomalyDetector(const std::string& onnx_path,
                                 const std::string& memory_bank_path,
                                 float threshold)
    : env_(ORT_LOGGING_LEVEL_ERROR, "Sehfeld"), threshold_(threshold)
{
    try {
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::wstring wpath(onnx_path.begin(), onnx_path.end());
        session_ = new Ort::Session(env_, wpath.c_str(), session_options_);
    } catch (const Ort::Exception& e) {
        std::cerr << "[AnomalyDetector] ORT: " << e.what() << "\n";
        return;
    }

    std::ifstream f(memory_bank_path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[AnomalyDetector] Cannot open memory bank\n";
        return;
    }
    int rows, cols;
    f.read((char*)&rows, sizeof(int));
    f.read((char*)&cols, sizeof(int));
    memory_bank_ = cv::Mat(rows, cols, CV_32F);
    f.read((char*)memory_bank_.data, rows * cols * sizeof(float));
    f.close();

    faiss_index_ = new faiss::IndexFlatL2(cols);
    faiss_index_->add(rows, (float*)memory_bank_.data);

    ready_ = true;
    std::cout << "[AnomalyDetector] Ready - " << rows << " patches\n";
}

AnomalyDetector::~AnomalyDetector()
{
    delete session_;
    delete faiss_index_;
}

bool AnomalyDetector::isReady() const { return ready_; }

DetectionResult AnomalyDetector::inspect(const cv::Mat& image,
                                          const std::string& image_path)
{
    DetectionResult result;
    result.image_path = image_path;
    result.timestamp  = currentTimestamp();
    result.frame_id   = ++frame_counter_;
    result.threshold  = threshold_;
    if (!ready_ || image.empty()) return result;

    cv::Mat blob     = preprocess(image);
    cv::Mat features = extractFeatures(blob);
    cv::Mat amap     = computeAnomalyMap(features);

    double minVal, maxVal;
    cv::minMaxLoc(amap, &minVal, &maxVal);
    cv::Mat amap_norm;
    amap.convertTo(amap_norm, CV_32F);
    if (maxVal > minVal)
        amap_norm = (amap_norm - (float)minVal) / (float)(maxVal - minVal);

    result.anomaly_score = (float)maxVal;
    result.is_defect     = result.anomaly_score > threshold_;

    // Absolute normalization — scores below threshold = blue, above = red
    cv::Mat amap_abs;
    float ceiling = threshold_ * 1.4f;
    amap.convertTo(amap_abs, CV_32F);
    amap_abs = amap_abs / ceiling;
    amap_abs = cv::min(amap_abs, 1.0f);
    amap_abs = cv::max(amap_abs, 0.0f);
    cv::Mat amap_8u;
    amap_abs.convertTo(amap_8u, CV_8U, 255.0);
    cv::resize(amap_8u, amap_8u, image.size());
    cv::applyColorMap(amap_8u, result.heatmap, cv::COLORMAP_JET);

    cv::Point maxLoc;
    cv::minMaxLoc(amap_norm, nullptr, nullptr, nullptr, &maxLoc);
    result.defect_location = cv::Point(
        (int)(maxLoc.x * (float)image.cols / amap_norm.cols),
        (int)(maxLoc.y * (float)image.rows / amap_norm.rows));

    return result;
}

cv::Mat AnomalyDetector::preprocess(const cv::Mat& image)
{
    cv::Mat rgb, resized;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32F, 1.0/255.0);

    std::vector<cv::Mat> ch(3);
    cv::split(resized, ch);
    ch[0] = (ch[0] - 0.485f) / 0.229f;
    ch[1] = (ch[1] - 0.456f) / 0.224f;
    ch[2] = (ch[2] - 0.406f) / 0.225f;

    cv::Mat blob(1 * 3 * 256 * 256, 1, CV_32F);
    int idx = 0;
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 256; h++)
            for (int w = 0; w < 256; w++)
                blob.at<float>(idx++) = ch[c].at<float>(h, w);
    int sz[4] = {1, 3, 256, 256};
    return blob.reshape(1, 4, sz);
}

cv::Mat AnomalyDetector::extractFeatures(const cv::Mat& blob)
{
    std::vector<int64_t> shape = {1, 3, 256, 256};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, (float*)blob.data, 1*3*256*256, shape.data(), shape.size());

    const char* in[]  = {"input"};
    const char* out[] = {"layer2_features", "layer3_features"};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in, &input, 1, out, 2);

    auto& t2 = outputs[0];
    auto shape_info2 = t2.GetTensorTypeAndShapeInfo();
    auto s2 = shape_info2.GetShape();
    int C2 = (int)s2[1], H2 = (int)s2[2], W2 = (int)s2[3];
    float* d2 = t2.GetTensorMutableData<float>();

    auto& t3 = outputs[1];
    auto shape_info3 = t3.GetTensorTypeAndShapeInfo();
    auto s3 = shape_info3.GetShape();
    int C3 = (int)s3[1], H3 = (int)s3[2], W3 = (int)s3[3];
    float* d3 = t3.GetTensorMutableData<float>();

    int H = H2, W = W2, C = C2 + C3;
    int sz[4] = {1, C, H, W};
    cv::Mat feat(4, sz, CV_32F);
    float* od = (float*)feat.data;

    for (int c = 0; c < C2; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                od[c*H*W + h*W + w] = d2[c*H2*W2 + h*W2 + w];

    for (int c = 0; c < C3; c++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float src_h = ((float)h + 0.5f) * H3 / H - 0.5f;
                float src_w = ((float)w + 0.5f) * W3 / W - 0.5f;

                src_h = std::max(0.0f, std::min(src_h, (float)(H3 - 1)));
                src_w = std::max(0.0f, std::min(src_w, (float)(W3 - 1)));

                int h0 = (int)src_h, w0 = (int)src_w;
                int h1 = std::min(h0 + 1, H3 - 1);
                int w1 = std::min(w0 + 1, W3 - 1);

                float dh = src_h - h0, dw = src_w - w0;

                float v00 = d3[c*H3*W3 + h0*W3 + w0];
                float v01 = d3[c*H3*W3 + h0*W3 + w1];
                float v10 = d3[c*H3*W3 + h1*W3 + w0];
                float v11 = d3[c*H3*W3 + h1*W3 + w1];

                od[(C2+c)*H*W + h*W + w] =
                    v00*(1-dh)*(1-dw) +
                    v01*(1-dh)*dw     +
                    v10*dh*(1-dw)     +
                    v11*dh*dw;
            }
        }
    }

    return feat.clone();
}

cv::Mat AnomalyDetector::computeAnomalyMap(const cv::Mat& features)
{
    if (!faiss_index_ || memory_bank_.empty())
        return cv::Mat::zeros(32, 32, CV_32F);

    int C = features.size[1], H = features.size[2], W = features.size[3];

    cv::Mat patches(H*W, C, CV_32F);
    const float* data = (const float*)features.data;
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            for (int c = 0; c < C; c++)
                patches.at<float>(h*W+w, c) = data[c*H*W + h*W + w];

    std::vector<faiss::idx_t> indices(H*W);
    std::vector<float>        dists(H*W);
    faiss_index_->search(H*W, (float*)patches.data, 1, dists.data(), indices.data());

    cv::Mat amap(H, W, CV_32F);
    for (int i = 0; i < H*W; i++)
        amap.at<float>(i/W, i%W) = std::sqrt(dists[i]);

    return amap;
}

std::string AnomalyDetector::currentTimestamp()
{
    time_t t = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&t));
    return std::string(buf);
}