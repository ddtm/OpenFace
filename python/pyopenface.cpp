#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <opencv2/core.hpp>

#include "detector.hpp"

namespace bp = boost::python;

void Detector_Run(Detector *detector, bp::object frame_obj) {
  PyArrayObject* frame_arr =
      reinterpret_cast<PyArrayObject*>(frame_obj.ptr());

  const int height = PyArray_DIMS(data_arr)[0];
  const int width = PyArray_DIMS(data_arr)[1];
  cv::Mat frame(cv::Size(width, height), CV_8UC1, PyArray_DATA(frame_arr),
                cv::Mat::AUTO_STEP);

  cv::Mat_<double> landmarks = detector->Run(frame);

  int landmarks_size = {landmarks.rows, landmarks.cols};
  PyObject * landmarks_obj = PyArray_SimpleNewFromData(
      2, &landmarks_size, NPY_DOUBLE, landmarks.data);
  bp::handle<> handle(landmarks_obj);
  bp::numeric::array landmarks_arr(handle);

  return landmarks_arr.copy();
}

BOOST_PYTHON_MODULE(pyopenface) {
  // numpy requires this
  import_array();
  
  bp::class_<Detector>("Detector", bp::no_init)
      .def("__init__", bp::make_constructor(&Detector::Create))
      .def("run", &Detector_Run);
}
