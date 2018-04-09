import os
import inspect

def getCallingFileName():
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    f = os.path.basename(filename)
    f = f[:f.rfind('.')]
    return f