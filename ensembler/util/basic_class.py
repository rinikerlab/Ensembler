import pickle
import io
from typing import Iterable, Sized, Union, Dict, Callable

def notImplementedERR():
    raise NotImplementedError("This function needs to be implemented in sympy")


class super_baseClass:
    """
    This class is a scaffold, containing functionality all classes should have.
    """

    name:str = "Unknown"

    def __name__(self) -> str:
        return str(self.name)

    def __getstate__(self):
        """
        preperation for pickling:
        remove the non trivial pickling parts
        """
        dict = self.__dict__

        keys = list(dict.keys())
        for key in keys:
            value = dict[key]

            #print(key, '\t', isinstance(dict[key], Callable) ,"\t", value, hasattr(value, "__getstate__"))
            if(isinstance(dict[key], Callable)):
                del dict[key]

        return dict

    def __setstate__(self, state):
        self.__dict__ = state

    """
    Methods
    """
    def save(self, path:Union[str, io.FileIO]=None)->str:
        """
        This method stores the Class as binary obj to a given path or fileBuffer.
        """
        if(isinstance(path, str)):
            bufferdWriter = open(path, "wb")
        elif(isinstance(path, io.BufferedWriter)):
            bufferdWriter =path
            path = bufferdWriter.name
        else:
            raise IOError("Please give as parameter a path:str or a File Buffer. To "+str(self.__class__)+".save")

        pickle.dump(obj=self, file=bufferdWriter)
        bufferdWriter.close()
        return path

    @classmethod
    def load(cls, path:Union[str, io.FileIO]=None)->object:
        """
        This method stores the Class as binary obj to a given path or fileBuffer.
        """
        if(isinstance(path, str)):
            bufferedReader = open(path, "rb")
        elif(isinstance(path, io.BufferedReader)):
            bufferedReader = path
        else:
            raise IOError("Please give as parameter a path:str or a File Buffer.")

        obj = pickle.load(file=bufferedReader)

        bufferedReader.close()

        return obj