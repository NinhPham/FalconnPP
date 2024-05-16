#include <FalconnPP.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace python {

namespace py = pybind11;

PYBIND11_MODULE(FalconnPP, m) { // Must be the same name with class FalconnPP
    py::class_<FalconnPP>(m, "FalconnPP")
        .def(py::init<const int&, const int&>(),  py::arg("n_points"), py::arg("n_features"))
//        .def("Index2Layers", &FalconnPP::Index2Layers)
//        .def("setQueryParam", &FalconnPP::setQueryParam)
        .def("setIndexParam", &FalconnPP::Index2Layers,
            py::arg("n_tables"), py::arg("n_proj"), py::arg("bucket_minSize"),
            py::arg("bucket_scale"), py::arg("iProbes"), py::arg("n_threads")
        )
        .def("set_qProbes", &FalconnPP::set_qProbes, py::arg("qProbes"))
        .def("set_threads", &FalconnPP::set_threads, py::arg("n_threads"))
        .def("clear", &FalconnPP::clear)
        .def("build2D", &FalconnPP::build2Layers, py::arg("dataset"))
        .def("query2D", &FalconnPP::query2Layers, py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false) // set default value = false
        .def("build", &FalconnPP::build2Layers_1D, py::arg("dataset"))
        .def("query", &FalconnPP::query2Layers_1D,
             py::arg("queries"), py::arg("n_neighbors"), py::arg("verbose") = false);

} // namespace FalconPP
} // namespace python
