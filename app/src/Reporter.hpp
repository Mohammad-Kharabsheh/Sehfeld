#pragma once
#include "AnomalyDetector.hpp"
#include <string>
#include <vector>

class Reporter {
public:
    static bool exportJSON(const DetectionResult& result,
                           const std::string& output_path);

    static bool exportBatchJSON(const std::vector<DetectionResult>& results,
                                const std::string& output_path);
};
