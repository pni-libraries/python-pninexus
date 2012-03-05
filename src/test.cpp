

#include<boost/python.hpp>

//include headerfiles for numpy
extern "C"{
#include<numpy/arrayobject.h>
}

#include<iostream>

using namespace boost::python;

void print_hello(){
    std::cout<<"hello world"<<std::endl;
}

template<typename T> void print_data(size_t n,T *data){
    for(size_t i=0;i<n;i++)
        std::cout<<data[i]<<" ";

    std::cout<<std::endl;
}

void ainfo(object &obj)
{
    PyArrayObject *ptr = (PyArrayObject *)obj.ptr();
    std::cout<<"Number of dimensions: "<<ptr->nd<<std::endl;
    size_t s = 1;
    for(size_t i=0;i<ptr->nd;i++){
        s *= ptr->dimensions[i];
        std::cout<<"Elements along "<<i<<" : "<<ptr->dimensions[i];
        std::cout<<std::endl;
    }
    std::cout<<"Total size: "<<s<<std::endl;
    
    //readout data
    PyArray_Descr *descr = ptr->descr;
    switch(PyArray_TYPE(obj.ptr())){
        case NPY_BOOL: std::cout<<"bool"<<std::endl;break;
        case NPY_BYTE: std::cout<<"byte"<<std::endl;break;
        case NPY_UBYTE: std::cout<<"unsigned byte"<<std::endl; break;
        case NPY_SHORT: std::cout<<"short"<<std::endl; break;
        case NPY_USHORT: std::cout<<"unsigned short"<<std::endl; break;
        case NPY_INT: std::cout<<"int"<<std::endl; break;
        case NPY_UINT: std::cout<<"unsigned int"<<std::endl; break;
        case NPY_LONG: print_data(s,(long *)PyArray_DATA(obj.ptr())); break;
        case NPY_ULONG: std::cout<<"unsigned long"<<std::endl; break;
        default:
                        std::cout<<"unknown type"<<std::endl;
    };

}

BOOST_PYTHON_MODULE(nxh5)
{
    def("print_hello",&print_hello);
    def("ainfo",&ainfo);
}

