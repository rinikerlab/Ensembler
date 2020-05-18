import os,sys
import unittest
import numpy as np
from numbers import Number
from collections.abc import Iterable

sys.path.append(os.path.dirname(__file__+"/.."))

from  ensembler.potentials._baseclasses import _potential1DCls, _potential2DCls, _potentialNDCls, _potentialNDMultiState


"""
TEST for Potentials ND
"""
class potentialNDCls(unittest.TestCase):
    """
    TEST for Potential inputs
    """

    def test_check_positions_float_type(self):
        # check single Float
        position = 1.0
        expected = position

        checked_pos = _potentialNDCls._check_positions_type_singlePos(
            position=position)

        #print(checked_pos)
        if (not isinstance(checked_pos, Number)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))

    def test_check_positions_npArray_type(self):
        # check nparray
        position = np.arange(1, 10)
        expected = [[pos] for pos in position]
        checked_pos = _potentialNDCls._check_positions_type_multiPos(
            positions=position)

        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_list_type(self):
        # check LIST[Float]
        position = [1.0, 2.0, 3.0]
        expected = np.array(position, ndmin=2)

        checked_pos = _potentialNDCls._check_positions_type_multiPos(
            positions=position)

        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_nDlist_type(self):
        position = [[1.0, 2.0, 3.0]]
        expected = np.array(position, ndmin=2)

        checked_pos = _potentialNDCls._check_positions_type_multiPos(
            positions=position)

        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_2Dlist_type(self):
        position = [[1.0, 2.0], [3.0, 4.0]]
        expected = np.array(position, ndmin=2)
        checked_pos = _potentialNDCls._check_positions_type_multiPos(
            positions=position)

        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

"""
TEST for Potentials 1D
"""
class potential1DCls(unittest.TestCase):

    """
    TEST for Potential inputs
    """
    def test_check_positions_float_type(self):
        #check single Float
        position = 1.0
        checked_pos = _potential1DCls._check_positions_type_multiPos(positions=position)

        #print(checked_pos)
        if(not isinstance(checked_pos, Iterable)):
            #print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - no list")
        elif(any([not isinstance(pos, Number) for pos in checked_pos])):
            #print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - not all list elements are float")

    def test_check_positions_npArray_type(self):
        #check nparray
        position = np.arange(1,10)
        checked_pos = _potential1DCls._check_positions_type_multiPos(positions=position)
        #print(checked_pos)

        if (not isinstance(checked_pos, Iterable)):
            #print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - no list")
        elif (any([not isinstance(pos, Number) for pos in checked_pos])):
            #print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - not all list elements are float")


    def test_check_positions_list_type(self):
        #check LIST[Float]
        position = [1.0, 2.0, 3.0]
        checked_pos = _potential1DCls._check_positions_type_multiPos(positions=position)
        #print(checked_pos)

        if (not isinstance(checked_pos, Iterable)):
            #print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - no list")
        elif (any([not isinstance(pos, Number) for pos in checked_pos])):
            #print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - not all list elements are float")


    def test_check_positions_nDlist_type(self):
        position = [[1.0, 2.0, 3.0]]
        expected = [1.0, 2.0, 3.0]
        checked_pos = _potential1DCls._check_positions_type_multiPos(positions=position)

        if (not isinstance(checked_pos, Iterable)):
            #print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - no list")
        elif (any([not isinstance(pos, Number) for pos in checked_pos])):
            #print(type(checked_pos), type(checked_pos[0]))
            raise Exception("The return Type has to be Iterable[Float] - not all list elements are float")

    def test_check_positions_2Dlist_type(self):
        position = [[1.0, 2.0], [3.0, 4.0]]
        expected = 'list dimensionality does not fit to potential dimensionality! len(list)=2 potential Dimensions 1'
        try:
            checked_pos = _potential1DCls._check_positions_type_multiPos(positions=position)
        except Exception as err:
            ##print(err.args)
            self.assertEqual(expected, err.args[0])
            #print("Found Err")
            return 0
        #print(checked_pos)
        #print("Did finish without error!")
        raise Exception("I expected an error here!")

"""
TEST for Potentials 2D
"""
class potential2DCls(unittest.TestCase):
    """
    TEST for Potential inputs
    """

    def test_check_position_1Dfloat_type(self):
        # check single Float
        position = 1.0
        expected = np.array(position, ndmin=2)

        try:
            checked_pos = _potential2DCls._check_positions_type(positions=position)
        except :
            #print("got error")
            return 0

        #print(checked_pos)
        #print("Did not get an Error!")
        exit(1)

    def test_check_position_2Dfloat_type_singlePos(self):
        # check single Float
        position = (1.0, 1.0)
        expected = np.array(position, ndmin=1)
        checked_pos = _potential2DCls._check_positions_type_singlePos(position=position)

        #print(checked_pos)
        #print("Did not get an Error!")
        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (len(checked_pos) != len(expected)):
            raise Exception("The returned Iterable has to have the length of 2 like the wanted dimensionality.\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))

        elif (any([not isinstance(dimPos, Number) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Number] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_2Dlist_type(self):
        # check LIST[Float]
        position = [(1.0, 2.0), (3.0, 4.0)]
        expected = np.array(position, ndmin=2)

        checked_pos = _potential2DCls._check_positions_type_multiPos(positions=position)

        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_nDlist_type(self):
        position = [[1.0, 2.0, 3.0]]

        try:
            checked_pos = _potential2DCls._check_positions_type(positions=position)
        except:
            #print("got error")
            return 0

        #print(checked_pos)
        #print("Did not get an Error!")
        exit(1)


"""
TEST for Potentials ND Multistate
"""
class potentialNDMultiStateCls(unittest.TestCase):
    """
    TEST for Potential inputs
    """

    potential:_potentialNDMultiState

    def setUp(self) -> None:
        self.potential=_potentialNDMultiState(nDim=-1, nStates=2)

    def test_check_positions_float_type(self):
        # check single Float
        position = 1.0
        expected = np.array([[1.0], [1.0]])
        #print(_potentialNDMultiState.nDim)
        checked_pos = self.potential._check_positions_type_singlePos(position=position)

        #print(checked_pos)
        self.assertListEqual(list(expected), list(checked_pos), msg="The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                            "\tgot: " + str(checked_pos))

    def test_check_positions_npArray_type(self):
        # check nparray
        position = np.arange(1, 10)
        expected = [[pos] for pos in position]
        checked_pos = self.potential._check_positions_type_multiPos(
            positions=position)

        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_list_type(self):
        # check LIST[Float]
        position = [1.0, 2.0, 3.0]
        expected = np.array(position, ndmin=2)

        checked_pos = self.potential._check_positions_type_multiPos(positions=position)

        #print(checked_pos)
        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

    def test_check_positions_nDlist_type(self):
        position = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        expected = np.array(position, ndmin=2)

        checked_pos = self.potential._check_positions_type_multiPos(
            positions=position)

        self.assertIsInstance(obj=checked_pos,cls=Iterable, msg="The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                            "\tgot: " + str(checked_pos))
        self.assertTrue(all([isinstance(dimPos, Iterable) for dimPos in checked_pos]), msg= "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                            "\tgot: " + str(checked_pos))
        self.assertTrue(all([all([isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos]), msg="The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                "\tgot: " + str(checked_pos))

    def test_check_positions_2Dlist_type(self):
        position = [[1.0, 2.0], [3.0, 4.0]]
        expected = np.array(position, ndmin=2)
        checked_pos = self.potential._check_positions_type_multiPos(
            positions=position)

        if (not isinstance(checked_pos, Iterable)):
            raise Exception("The return Type has to be Iterable[Iterable[Number]] - no list. Missing top layer list!\n"
                            "\texpected: " + str(expected) + "\n"
                                                             "\tgot: " + str(checked_pos))
        elif (any([not isinstance(dimPos, Iterable) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing dimension layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))
        elif (any([any([not isinstance(pos, Number) for pos in dimPos]) for dimPos in checked_pos])):
            raise Exception(
                "The return Type has to be Iterable[Iterable[Number]] - no list. Missing number layer list!\n"
                "\texpected: " + str(expected) + "\n"
                                                 "\tgot: " + str(checked_pos))

class perturbedPotentialNDMultiStateCls(unittest.TestCase):
    pass
