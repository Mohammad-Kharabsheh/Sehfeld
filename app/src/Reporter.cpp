#include "Reporter.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

static std::string escapeJSON(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"')       out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

bool Reporter::exportJSON(const DetectionResult& r,
                          const std::string& output_path)
{
    std::ofstream f(output_path);
    if (!f.is_open()) return false;
    f << std::fixed << std::setprecision(4);
    f << "{\n";
    f << "  \"frame_id\"      : " << r.frame_id << ",\n";
    f << "  \"timestamp\"     : \"" << escapeJSON(r.timestamp) << "\",\n";
    f << "  \"image_path\"    : \"" << escapeJSON(r.image_path) << "\",\n";
    f << "  \"status\"        : \"" << (r.is_defect ? "DEFECT" : "PASS") << "\",\n";
    f << "  \"anomaly_score\" : " << r.anomaly_score << ",\n";
    f << "  \"threshold\"     : " << r.threshold << ",\n";
    f << "  \"defect\" : {\n";
    f << "    \"location_x\"  : " << r.defect_location.x << ",\n";
    f << "    \"location_y\"  : " << r.defect_location.y << "\n";
    f << "  }\n";
    f << "}\n";
    f.close();
    return true;
}

bool Reporter::exportBatchJSON(const std::vector<DetectionResult>& results,
                               const std::string& output_path)
{
    std::ofstream f(output_path);
    if (!f.is_open()) return false;
    int defects = 0;
    for (const auto& r : results)
        if (r.is_defect) defects++;
    f << std::fixed << std::setprecision(4);
    f << "{\n";
    f << "  \"total\"   : " << results.size() << ",\n";
    f << "  \"defects\" : " << defects << ",\n";
    f << "  \"passed\"  : " << (results.size() - defects) << ",\n";
    f << "  \"results\" : [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        f << "    {\n";
        f << "      \"frame_id\"      : " << r.frame_id << ",\n";
        f << "      \"image_path\"    : \"" << escapeJSON(r.image_path) << "\",\n";
        f << "      \"status\"        : \"" << (r.is_defect ? "DEFECT" : "PASS") << "\",\n";
        f << "      \"anomaly_score\" : " << r.anomaly_score << ",\n";
        f << "      \"timestamp\"     : \"" << escapeJSON(r.timestamp) << "\"\n";
        f << "    }";
        if (i < results.size() - 1) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
    f.close();
    return true;
}