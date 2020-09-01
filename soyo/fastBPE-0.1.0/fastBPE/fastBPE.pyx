# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "fastBPE.hpp" namespace "fastBPE":
    cdef cppclass BPEApplyer:
        BPEApplyer(const string& codes_path, const string& vocab_path)
        vector[string] apply(vector[string]& sentences)

cdef class fastBPE:
    cdef BPEApplyer* c_obj

    def __dealloc__(self):
        del self.c_obj

    def __init__(self, codes_path, vocab_path=""):
        self.c_obj = new BPEApplyer(codes_path.encode(), vocab_path.encode())

    def apply(self, sentences):
        cdef vector[string] s = [x.encode() for x in sentences]
        cdef vector[string] res = self.c_obj.apply(s)
        return [x.decode() for x in res]
