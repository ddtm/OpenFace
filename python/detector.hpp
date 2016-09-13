#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <LandmarkCoreIncludes.h>

using std::unique_ptr;

class Detector {
public:
  static Detector * Create(const char *binary_path);
  std::tuple<cv::Mat_<double>, cv::Mat_<int>> Run(
      const cv::Mat &input_frame, bool use_full_region = false);

private:
  Detector(LandmarkDetector::FaceModelParameters &det_parameters,
           LandmarkDetector::CLNF &clnf_model,
           cv::CascadeClassifier &classifier,
           dlib::frontal_face_detector &face_detector_hog);

  cv::Rect_<double> DetectFace(const cv::Mat &grayscale_frame);
  cv::Mat_<int> GetVisibilities();

  LandmarkDetector::FaceModelParameters det_parameters_;
  LandmarkDetector::CLNF clnf_model_;
  cv::CascadeClassifier classifier_;
  dlib::frontal_face_detector face_detector_hog_;

  cv::Mat_<uchar> grayscale_frame_;
};
