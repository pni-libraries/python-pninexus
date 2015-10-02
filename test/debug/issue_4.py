import pni.io.nx.h5 as nexus

f = nexus.create_file("issue_4.nxs", True)
rt = f.root()
fd = rt.create_field("my", dtype="int32")
