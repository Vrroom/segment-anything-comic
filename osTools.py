import os
import os.path as osp

def mkdir(path) : 
    if osp.exists(path) : 
        return 
    try : 
        os.mkdir(path) 
    except FileNotFoundError : 
        parentPath, _ = osp.split(path)
        mkdir(parentPath)
        os.mkdir(path)

def getBaseName(fullName) : 
    return osp.splitext(osp.split(fullName)[1])[0]

def zipDirs (dirList) :
    """ 
    A common operation is to get SVGs
    or graphs from different directories
    and match them up and perform 
    some operations on them. 

    Parameters
    ----------
    dirList : list
        List of directories to zip.
    """
    filesInDirList = list(map(listdir, dirList))
    return zip(*filesInDirList)

def listdir (path) :
    """
    Convenience function to get 
    full path details while calling os.listdir

    Also ensures that the order is always the same.

    Parameters
    ----------
    path : str
        Path to be listed.
    """
    paths = [osp.join(path, f) for f in os.listdir(path)]
    paths.sort()
    return paths

def allfiles (directory) :
    """ List full paths of all files/directory in directory """
    for f in listdir(directory) : 
        yield f 
        if osp.isdir(f) : 
            yield from allfiles(f)

def allFilesWithSuffix(directory, suffix) : 
    """ List full paths of all files that end with suffix """ 
    return filter(lambda x : x.endswith(suffix), allfiles(directory))

def allDirs (directory) : 
    """ List all directories within this directory """
    return filter(osp.isdir, listdir(directory))

def allFiles (directory) :
    """ List all files that are not directories in this directory """
    return filter(lambda x : not osp.isdir(x), listdir(directory))

def relpathToAbsPath (x) : 
    """
    Returns the absolute path for a relative path. Can be convenient in some scenarios.
    """
    return osp.normpath(osp.join(osp.split(osp.abspath(__file__))[0], x))

