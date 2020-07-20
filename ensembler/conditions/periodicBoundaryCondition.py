import numpy as np

from ensembler.conditions._conditions import _conditionCls
from ensembler.util.ensemblerTypes import system as systemType, Iterable

class periodicBoundaryCondition(_conditionCls):
    """
        ..autoclass:: periodicBoundaryCondition
            This class allows to enable sampling in mirror images and projects the coordinates to the restricted space.
    """

    lowerbounds:Iterable
    higherbounds:Iterable

    def __init__(self, boundary:Iterable, system:systemType=None):
        self._parse_boundary(boundary)
        if(system != None):
            self.system = system
            self.nDim = system.nDim
            self.nStates = system.nStates

    def __str__(self)->str:
        msg = "Periodic Boundary Condition\n"
        msg += "\tDimensions: "+str(self.nDim)+"\n"
        #msg += "\tStates: "+str(self.nStates)+"\n"
        msg += "\n"
        msg += "\tapply every step: "+str(self.nDim)+"\n"
        msg += "\tHigher bounds: "+str(self.higherbounds)+"\n"
        msg += "\tLower bounds: "+str(self.lowerbounds)+"\n"
        return msg

    def apply(self):
        """
        not covering all cases!!!
        TODO: reimplement!
        :return:
        """

        if(self.nDim == 1):
            new_current_position = self.system._currentPosition
            if (new_current_position < self.lowerbounds):
                new_current_position = self.higherbounds - (self.lowerbounds - new_current_position)
            elif (new_current_position > self.higherbounds):
                new_current_position = self.lowerbounds + (new_current_position - self.higherbounds)

            self.system._currentPosition = new_current_position

        else:
            new_current_position = []
            for dim_pos, dimlBound, dimhBound in zip(self.system._currentPosition, self.lowerbounds, self.higherbounds):
                if(dim_pos < dimlBound):
                    new_current_position.append(dimhBound - (dimlBound - dim_pos))
                    #new_current_position.append(np.subtract(dimhBound, np.subtract(dimlBound, dim_pos%dimhBound)))
                elif(dim_pos > dimhBound):
                    new_current_position.append(dimlBound + (dim_pos- dimhBound))
                    #new_current_position.append(np.add(dimlBound, np.subtract(dim_pos%dimlBound, dimhBound)))
                else:
                    new_current_position.append(dim_pos)
            self.system._currentPosition = new_current_position
        #print("pBC2: ", self.system._currentPosition)

    def _parse_boundary(self, boundary):  #Todo: really needed?
        if(isinstance(boundary, Iterable)):
            if (isinstance(boundary[0], Iterable)):
                self.higherbounds = np.array(list(map(max, boundary )))
                self.lowerbounds = np.array(list(map(min, boundary )))

            else:
                self.higherbounds = max(boundary)
                self.lowerbounds = min(boundary)
        return True