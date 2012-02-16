#This module contains classes to check for the existance of a 
#particular program - this checkers can be used to ensure that a 
#particular program is installed before the build.

def CheckProgram(context,prg):
    context.Message("checking for program "+prg+" ...")
    act = context.env.Action(prg)
    result = context.TryAction(act)
    context.Result(result[0])
    return result[0]