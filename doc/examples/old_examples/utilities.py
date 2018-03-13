import pni.io.nx.h5 as nexus

def get_entry_list(nexus_file):
    """
    Returns a dictionary of all entries in the file. The key is the name of
    each entry instance stored in the file. This function can come in quite
    handy if a file contains several entries. 

    Arg:
        nexus_file (nxfile)          reference to a NeXus file

    Returns:
        Dictionary with instances of NXentry below the root node. 
    """
    entries = {}
    root = nexus_file.root()

    #simple iteration is enough for here - we just need the direct children 
    #of the root group
    for g in root:

        #if the child object is a group and of class NXentry it will be 
        #added to the dictionary 
        if isinstance(g,nexus.nxgroup) and nexus.is_class(g,"NXentry"):
            entries[g.name] = g

    return entries 


def plot_file(nexus_file):
    """
    Plot the content of a NeXus data file by utilizing the content of the 
    NXdata instance stored below each entry. 
    """
    pass

def write_everything(obj):
    return isinstance(obj,nexus.nxfield) or \
            isinstance(obj,nexus.nxattribute)
