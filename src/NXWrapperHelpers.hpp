#ifndef __NXWRAPPERHELPERS_HPP__
#define __NXWRAPPERHELPERS_HPP__

#include<pni/utils/Types.hpp>
#include<pni/utils/Shape.hpp>

#include<boost/python/list.hpp>
using namespace pni::utils;
using namespace boost::python;

/*! \brief create string from type id

Helper function that creatds a numpy type code string from a pni::utils::TypeID.
\param tid type id from pniutils
\return NumPy typecode
*/
String typeid2str(const TypeID &tid);

/*! \brief create list from shape

Creates a Python list from a Shape object. The length of the list corresponds to
the number of dimension in the Shape object. The lists elements are the numbers
of elements along each dimension.
\param s shape object
\return python list with 
*/
list Shape2List(const Shape &s);

/*! \brief list to Shape conversion

Converts a Python list to a Shape object. The length of the list is interpreted
as the number of dimensions and each element of the list as the number of
elements along a particular dimension.
\param l list object
\return Shape object
*/
Shape List2Shape(const list &l);


#endif
