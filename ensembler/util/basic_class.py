"""
Module: basic_class
    This file is giving the basic scaffold for saving & loading any Ensembler class with pickle.
"""

import io
import pickle
from typing import Union, Callable


def notImplementedERR():
    raise NotImplementedError("This function needs to be implemented in sympy")


class _baseClass:
    """
    This class is a scaffold, containing functionality all classes should have.
    """
    name: str = "Unknown"
    _verbose:bool =False

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
            if (isinstance(dict[key], Callable)):
                del dict[key]

        return dict

    def __setstate__(self, state):
        self.__dict__ = state

    """
    Attributes
    """
    @property
    def verbose(self)->bool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose:bool):
        self._verbose=verbose

    """
    Methods
    """
    def save(self, path: Union[str, io.FileIO] = None) -> str:
        """
        This method stores the Class as binary obj to a given path or fileBuffer.
        """
        if (isinstance(path, str)):
            bufferdWriter = open(path, "wb")
        elif (isinstance(path, io.BufferedWriter)):
            bufferdWriter = path
            path = bufferdWriter.name
        else:
            raise IOError("Please give as parameter a path:str or a File Buffer. To " + str(self.__class__) + ".save")

        pickle.dump(obj=self, file=bufferdWriter)
        bufferdWriter.close()
        return path

    @classmethod
    def load(cls, path: Union[str, io.FileIO] = None) -> object:
        """
        This method stores the Class as binary obj to a given path or fileBuffer.
        """
        if (isinstance(path, str)):
            bufferedReader = open(path, "rb")
        elif (isinstance(path, io.BufferedReader)):
            bufferedReader = path
        else:
            raise IOError("Please give as parameter a path:str or a File Buffer.")

        obj = pickle.load(file=bufferedReader)

        bufferedReader.close()

        return obj
