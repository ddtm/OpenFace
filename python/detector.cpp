#include "detector.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <LandmarkCoreIncludes.h>

#include <dlib/image_processing/frontal_face_detector.h>

using std::cout;
using std::string;
using std::vector;

Detector * Detector::Create(const char *binary_path) {
  vector<string> arguments;
  arguments.push_back(binary_path);

  LandmarkDetector::FaceModelParameters det_parameters(arguments);
  det_parameters.track_gaze = true;
  // No need to validate detections, as we're not doing tracking.
  det_parameters.validate_detections = false;

  // Grab camera parameters, if they are not defined 
  // (approximate values will be used).
  float fx = 0, fy = 0, cx = 0, cy = 0;
  int device = -1;
  // Get camera parameters
  LandmarkDetector::get_camera_params(device, fx, fy, cx, cy, arguments);

  // If cx (optical axis centre) is undefined will use the image size/2 as 
  // an estimate.
  bool cx_undefined = false;
  bool fx_undefined = false;
  if (cx == 0 || cy == 0) {
    cx_undefined = true;
  }
  if (fx == 0 || fy == 0) {
    fx_undefined = true;
  }

  // The modules that are being used for tracking.
  LandmarkDetector::CLNF clnf_model(det_parameters.model_location);
  
  cv::CascadeClassifier classifier(det_parameters.face_detector_location);
  dlib::frontal_face_detector face_detector_hog =
      dlib::get_frontal_face_detector();

  return new Detector(
      det_parameters, clnf_model, classifier, face_detector_hog);
}

Detector::Detector(LandmarkDetector::FaceModelParameters &det_parameters,
                   LandmarkDetector::CLNF &clnf_model,
                   cv::CascadeClassifier &classifier,
                   dlib::frontal_face_detector &face_detector_hog) :
det_parameters_(std::move(det_parameters)), clnf_model_(std::move(clnf_model)),
classifier_(std::move(classifier)), 
face_detector_hog_(std::move(face_detector_hog)) {}

bool CompareRect(cv::Rect_<double> r1, cv::Rect_<double> r2) {
  return r1.height < r2.height;
}

cv::Rect_<double> Detector::DetectFace(const cv::Mat &grayscale_frame) {
  vector<cv::Rect_<double>> face_detections;

  if(det_parameters_.curr_face_detector ==
     LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR) {
    vector<double> confidences;
    cout << "  DetectFacesHOG" << std::endl;
    LandmarkDetector::DetectFacesHOG(
        face_detections, grayscale_frame, face_detector_hog_, confidences);
  } else {
    cout << "  DetectFaces" << std::endl;
    LandmarkDetector::DetectFaces(
        face_detections, grayscale_frame, classifier_);
  }
  
  // Finding the biggest face among the detected ones.
  cout << "  Find biggest face" << std::endl;
  if (face_detections.empty()) {
    throw std::invalid_argument("No faces detected");
  }
  cv::Rect_<double> face_l = *max_element(
      face_detections.begin(), face_detections.end(), CompareRect);

  // Return the biggest face.
  return face_l;
}

cv::Mat_<int> Detector::GetVisibilities() {
  int idx = clnf_model_.patch_experts.GetViewIdx(clnf_model_.params_global, 0);
  return clnf_model_.patch_experts.visibilities[0][idx];
}

std::tuple<cv::Mat_<double>, cv::Mat_<int>> Detector::Run(
    const cv::Mat &input_frame, bool use_full_region) {
  cv::Mat_<float> depth_image;
  cv::cvtColor(input_frame, grayscale_frame_, CV_BGR2GRAY);

  // Detect faces in the image.
  cout << "DetectFace" << std::endl;
  cv::Rect_<double> face_rect(
      0, 0, grayscale_frame_.cols, grayscale_frame_.rows);
  if (!use_full_region) {
    try {
      face_rect = DetectFace(grayscale_frame_);
    } catch (...) {
      throw std::runtime_error("Unable to detect face in the frame");
    }
  }

  // Detect landmarks for the detected face.
  cout << "DetectLandmarksInImage" << std::endl;
  bool success = LandmarkDetector::DetectLandmarksInImage(
      grayscale_frame_, depth_image, face_rect, 
      clnf_model_, det_parameters_);
  if (!success) {
    throw std::runtime_error("Unable to detect landmarks");
  }

  cv::Mat_<double> landmarks_2d = clnf_model_.detected_landmarks;
  landmarks_2d = landmarks_2d.reshape(1, 2);

  cv::Mat_<int> visibilities = GetVisibilities();

  return std::make_tuple(landmarks_2d, visibilities);
}
