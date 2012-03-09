//helper functions to create wrappers

#include <boost/python/extract.hpp>
#include <boost/python/slice.hpp>
#include "NXWrapperHelpers.hpp"

//-----------------------------------------------------------------------------
String typeid2str(const TypeID &tid)
{
    if(tid == TypeID::STRING) return "string";
    if(tid == TypeID::UINT8) return "uint8";
    if(tid == TypeID::INT8)  return "int8";
    if(tid == TypeID::UINT16) return "uint16";
    if(tid == TypeID::INT16)  return "int16";
    if(tid == TypeID::UINT32) return "uint32";
    if(tid == TypeID::INT32)  return "int32";
    if(tid == TypeID::UINT64) return "uint64";
    if(tid == TypeID::INT64) return "int64";

    if(tid == TypeID::FLOAT32) return "float32";
    if(tid == TypeID::FLOAT64) return "float64";
    if(tid == TypeID::FLOAT128) return "float128";

    if(tid == TypeID::COMPLEX32) return "complex64";
    if(tid == TypeID::COMPLEX64) return "complex128";
    if(tid == TypeID::COMPLEX128) return "complex256";

    return "none";
}

//-----------------------------------------------------------------------------
list Shape2List(const Shape &s){
    list l;

    if(s.rank() == 0) return l;

    for(size_t i=0;i<s.rank();i++) l.append(s[i]);

    return l;

}

//-----------------------------------------------------------------------------
Shape List2Shape(const list &l){
    long size = len(l);
    Shape s(size);

    for(ssize_t i=0;i<size;i++){
        s.dim(i,extract<size_t>(l[i]));
    }

    return s;
}

//------------------------------------------------------------------------------
NXSelection create_selection(const tuple &t,const NXField &field)
{
    //obtain a selection object
    NXSelection selection = field.selection();

    //as the selection has the same rank as the field it belongs to we
    //can use this here for checking the length of the tupl
    if(len(t) != selection.shape().rank())
    {
        std::cerr<<"number of dimensions does not macht array"<<std::endl;
        //raise an exception here
    }

    for(size_t i=0;i<len(t);i++){
        extract<size_t> index(t[i]);

        if(index.check()){
            selection.offset(i,index);
            selection.shape(i,1);
            selection.stride(i,1);
            continue;
        }

        extract<slice> s(t[i]);
        if(s.check()){
            //now we have to investigate the components of the 
            //slice
            ssize_t start;
            extract<size_t> __start(s().start());
            if(__start.check())
                start = __start();
            else
                start = 0;
           
            ssize_t step;
            extract<size_t> __step(s().step());
            if(__step.check())
                step = __step();
            else
                step = 1;

            ssize_t stop;
            extract<ssize_t> __stop(s().stop());
            if(__stop.check())
                stop = __stop();
            else
                stop = field.shape().dim(i);

            //configure the selection
            selection.offset(i,start);
            selection.stride(i,step);
            
            ssize_t res = (stop-start)%step;
            selection.shape(i,(stop-start-res)/step);
        }

        //here we would need code to manage ellipses - that is still missing
    }

    return selection;

}
