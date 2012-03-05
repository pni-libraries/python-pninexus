

#include<boost/python.hpp>

#include<iostream>

using namespace boost::python;

void print_hello(){
    std::cout<<"hello world"<<std::endl;
}

BOOST_PYTHON_MODULE(pninx)
{
    def("print_hello",&print_hello);
}

