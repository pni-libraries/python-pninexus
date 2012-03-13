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
Shape Tuple2Shape(const tuple &t)
{
    long size = len(t);
    Shape s(size);

    for(size_t i=0;i<size;i++){
        s.dim(i,extract<size_t>(t[i]));
    }
    return s;
}

//------------------------------------------------------------------------------
NXSelection create_selection(const tuple &t,const NXField &field)
{
    //obtain a selection object
    NXSelection selection = field.selection();

    //the number of elements in the tuple must not be equal to the 
    //rank of the field. This is due to the fact that the tuple can contain
    //one ellipsis which spans over several dimensions.

    bool has_ellipsis = false;
    size_t ellipsis_size = 0;
    if(len(t) > selection.shape().rank()){
        //with or without ellipsis something went wrong here
        ShapeMissmatchError error;
        error.issuer("NXSelection create_selection(const tuple &t,"
                "const NXField &field)");
        error.description("Tuple with indices, slices, and ellipsis is "
                "longer than the rank of the field - something went wrong"
                "here");
        throw(error);
    }
    else if(len(t) != selection.shape().rank())
    {
        //here we have to fix the size of an ellipsis
        ellipsis_size = selection.shape().rank()-(len(t)-1);
    }

    /*this loop has tow possibilities:
    -> there is no ellipse and the rank of the field is larger than the size of
       the tuple passed. In this case an IndexError will occur. In this case we 
       know immediately that a shape error occured.
    -> 
    */
    for(size_t i=0,j=0;i<selection.shape().rank();i++,j++){
        //manage a single index
        extract<size_t> index(t[j]);

        if(index.check()){
            selection.offset(i,index);
            selection.shape(i,1);
            selection.stride(i,1);
            continue;
        }

        //manage a slice
        extract<slice> s(t[j]);
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
            continue;
        }

        //manage an ellipse
        const object &o = t[j];
        if(Py_Ellipsis != o.ptr())
        {
            std::cerr<<"not an ellipsis ..."<<std::endl;
            //raise an exception here
        }
        //assume here that the object is an ellipsis - this is a bit difficult
        //to handle as we do not know over how many 
        if(!has_ellipsis){
            has_ellipsis = true;
            while(i<j+ellipsis_size){
                selection.stride(i,1);
                selection.offset(i,0);
                i++;
            }
        }else{
            std::cerr<<"only one ellipsis is allowed per selection!"<<std::endl;
            //raise an exception here
        }
    }

    //once we are done with looping over all elemnts in the tuple we need 
    //to adjust the selection to take into account an ellipsis
    if((ellipsis_size) && (!has_ellipsis)){
        ShapeMissmatchError error;
        error.issuer("NXSelection create_selection(const tuple &t,const "
                "NXField &field)");
        error.description("Selection rank does not match field rank");
        throw(error);
    }

    return selection;

}
