#include <iostream>

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <opencv2/core.hpp>

#include "detector.hpp"

namespace bp = boost::python;

bp::object Detector_Run(Detector *detector, bp::object frame_obj,
                        bool use_full_rect = false) {
  PyArrayObject* frame_arr =
      reinterpret_cast<PyArrayObject*>(frame_obj.ptr());

  const int height = PyArray_DIMS(frame_arr)[0];
  const int width = PyArray_DIMS(frame_arr)[1];
  cv::Mat frame(cv::Size(width, height), CV_8UC3, PyArray_DATA(frame_arr),
                cv::Mat::AUTO_STEP);

  auto detection_results = detector->Run(frame, use_full_rect);
  cv::Mat_<double> landmarks = std::get<0>(detection_results);
  cv::Mat_<int> visibilities = std::get<1>(detection_results);
  std::cout << visibilities.rows << " " << visibilities.cols << std::endl;

  long int landmarks_size[2] = {landmarks.rows, landmarks.cols};
  PyObject * landmarks_obj = PyArray_SimpleNewFromData(
      2, landmarks_size, NPY_DOUBLE, landmarks.data);
  bp::handle<> landmarks_handle(landmarks_obj);
  bp::numeric::array landmarks_arr(landmarks_handle);

  long int visibilities_size = visibilities.rows;
  PyObject * visibilities_obj = PyArray_SimpleNewFromData(
      1, &visibilities_size, NPY_INT, visibilities.data);
  bp::handle<> visibilities_handle(visibilities_obj);
  bp::numeric::array visibilities_arr(visibilities_handle);

  return bp::make_tuple(landmarks_arr.copy(), visibilities_arr.copy());
}

BOOST_PYTHON_FUNCTION_OVERLOADS(
    Detector_Run_overloads, Detector_Run, 2, 3)

BOOST_PYTHON_MODULE(pyopenface) {
  bp::numeric::array::set_module_and_type("numpy", "ndarray");
  // numpy requires this
  import_array();
  
  bp::class_<Detector>("Detector", bp::no_init)
      .def("__init__", bp::make_constructor(&Detector::Create))
      .def("run", &Detector_Run, Detector_Run_overloads(
          bp::args("self", "frame", "use_full_rect"),
          "Run detector on frame"));
}
