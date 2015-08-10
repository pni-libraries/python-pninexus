Nexus and XML 
=============

NeXus provides easy means to acces the data stored in NeXus files even if the
structure of a NeXus file can quickly become rather complicated. 
However, the complexity of such a file can be quite painfull during file
creation. 

A NeXus XLM primer 
------------------



From XML to NeXus
-----------------

Creating NeXus files can be a quite challenging process simply because NeXus
allows us so much more information to be stored in a file in a structured way. 
Even if the basic structure of a file remains the same, in many cases small
changes have to be made to adopt a file structure to the current needs of 
a user. It is therefore hard, if not impossible, to encode the structure of the
file directly while at the same time keep the code maintainable. 

A more reasonable approach would be to describe the basic structure of a file 
along with some (semi) static metadata in a textual form and write generic code
which interprets this textual representation from which it will create the
NeXus file. 
Python is in particular famous for its many templating engines and thus a 
quite reasonable candidate to go in this direction. 



From NeXus to XML
-----------------

In some situations the structure of NeXus file has to be made aware 

