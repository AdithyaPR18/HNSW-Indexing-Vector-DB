#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "hnsw.h"

namespace py = pybind11;

// Thin adapter so callers can pass numpy arrays directly
class PyHNSWIndex {
public:
    PyHNSWIndex(int dim, int M, int ef_construction, unsigned int seed)
        : idx_(dim, M, ef_construction, seed) {}

    int insert(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
        auto buf = arr.request();
        if (buf.ndim != 1 || buf.shape[0] != idx_.dim()) {
            throw std::invalid_argument(
                "Expected 1-D array of length " + std::to_string(idx_.dim()));
        }
        float* ptr = static_cast<float*>(buf.ptr);
        std::vector<float> vec(ptr, ptr + idx_.dim());
        return idx_.insert(vec);
    }

    // Returns list of (distance, id) tuples, nearest first
    std::vector<std::pair<float, int>>
    search(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
           int k,
           int ef) {
        auto buf = arr.request();
        if (buf.ndim != 1 || buf.shape[0] != idx_.dim()) {
            throw std::invalid_argument(
                "Expected 1-D array of length " + std::to_string(idx_.dim()));
        }
        float* ptr = static_cast<float*>(buf.ptr);
        std::vector<float> vec(ptr, ptr + idx_.dim());
        auto results = idx_.search(vec, k, ef);
        return std::vector<std::pair<float, int>>(results.begin(), results.end());
    }

    void remove(int id) { idx_.remove(id); }
    int  size()  const  { return idx_.size(); }
    int  dim()   const  { return idx_.dim(); }

private:
    HNSWIndex idx_;
};

PYBIND11_MODULE(hnsw_index, m) {
    m.doc() = "HNSW approximate nearest neighbour index (C++ core)";

    py::class_<PyHNSWIndex>(m, "HNSWIndex")
        .def(py::init<int, int, int, unsigned int>(),
             py::arg("dim"),
             py::arg("M")               = 16,
             py::arg("ef_construction") = 200,
             py::arg("seed")            = 42,
             R"doc(
Create a new HNSW index.

Parameters
----------
dim : int
    Dimensionality of the vectors.
M : int
    Max number of bidirectional connections per node per layer.
    16 is a good default; higher M = better recall, slower build.
ef_construction : int
    Beam width during graph construction. Higher = better recall, slower build.
seed : int
    RNG seed for reproducible layer assignment.
)doc")
        .def("insert", &PyHNSWIndex::insert,
             py::arg("vec"),
             "Insert a float32 numpy array. Returns the internal node id.")
        .def("search", &PyHNSWIndex::search,
             py::arg("query"),
             py::arg("k"),
             py::arg("ef") = -1,
             "Return the k approximate nearest neighbours as [(dist, id), ...].")
        .def("remove", &PyHNSWIndex::remove,
             py::arg("id"),
             "Lazily delete a node by id.")
        .def("__len__", &PyHNSWIndex::size)
        .def_property_readonly("dim", &PyHNSWIndex::dim);
}
