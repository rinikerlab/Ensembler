from ensembler.system import system as sys
from ensembler.conditions._conditions import Condition

from scipy import constants as const
from scipy.stats import boltzmann

class thermostat(Condition):
    """
    ..autoclass: Thermostat
        This is the parent class of Thermostats.
        The apply function of this class is not implemented and needs to be overwritten by each subclass.
    """

    _currentTemperature:float
    system:sys #system
    verbose:bool = False

    def __init__(self, system:sys, tau:float, ):
        self.system = system
        self.tau = tau


class boltzmanTempBath(thermostat):
    pass
class berendsenThermostate(thermostat):
    """
    ..autoclass: berendsen Thermostat

        reference: Molecular dynamics with coupling to an external bath; H.J.C. Berendsen
    """

    def __init__(self, tau:float, dt:float, MConstraintsDims:int=1, system:sys=None ):
        self._lambda:float=1   #scaling factor of velocities
        self._current_temperatur = 1
        self.tau = tau
        self.dt = dt
        self.M = MConstraintsDims

        if(system != None):
            self.system = system

    def apply(self):
        self._calculate_current_temperature()
        self._calculate_scaling_factor()
        self._rescale_velocities()
        
        if(self.verbose):
            print("THERMOSTAT: get to temp: ", self.system.temperature,"\n"
                  'THERMOSTAT: tot_kin: ', self.system.totKin(),"\n"
                  "THERMOSTAT: curr temp: ", self._current_temperatur,"\n"
                  "THERMOSTAT: lambda: ", self._lambda,"\n"
                  "THERMOSTAT: current_Velocity: ", self.system._currentVelocities, "\n"
                  "\n")

    def _rescale_velocities(self):
        orig_vels = self.system._currentVelocities
        #new_vels = [self._lambda * velocity for velocity in orig_vels]
        new_vels = self._lambda * orig_vels if(self._lambda * orig_vels) else 0.0001    #do not allow 0 vel
        self.system._currentVelocities = new_vels

    def _calculate_current_temperature(self):
        """
        autofunction: _calculate current Temperature (eq. 32,33)
        :return:
        """
        #M = constraints
        N=self.system.nparticles * const.Avogadro
        self._current_temperatur = (2/(3*N-self.M-3))*self.system.totKin()*N

    def _calculate_scaling_factor(self):
        """
        autofunction: _calculate_scaling_factor
            (eq.34)
        :return:
        """
        T_t = (self._current_temperatur if (self._current_temperatur!=0) else 0.000001)
        self._lambda = (1+(self.dt/self.tau)*((self.system.temperature/T_t)-1))**0.5
