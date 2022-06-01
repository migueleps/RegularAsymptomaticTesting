"""
Contains various infectiousness profiles implemented in Numba that can be plugged into covasim.
"""

# requires the code that is currently in the time-varying infectiousness branch of covasim
from covasim.infection import InfectionDynamics
import covasim.defaults as cvd
from covasim.settings import options as cvo
import numpy as np
from numpy import float32, nan
import numba as nb
import matplotlib.pyplot as plt
import scipy.stats as ss

# Set dtypes -- note, these cannot be changed after import since Numba functions are precompiled
nbbool  = nb.bool_
nbint   = cvd.nbint
nbfloat = cvd.nbfloat

# TODO it might be possible to move the dictionary lookups into Numba to make it all faster
# Just need to figure out typed dictionaries...
# https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#dict

@nb.njit(              (nbint,  nbfloat[:],     nbfloat[:],     nbfloat[:], nbint))
def get_infectious_ages(t,      time_exposed,   time_recovered, time_dead,  upper_limit):
    """Rapidly computes the infectious age of individuals

    If an individual is susceptible, recovered, or dead, then their infectious age is returned as -1
    Which then gets translated into flaot32(np.nan) when the dictionary lookup is performed
    """

    infectious_ages = t - time_exposed
    infectious_ages[t > time_recovered] = -1
    infectious_ages[t > time_dead] = -1
    infectious_ages[infectious_ages > upper_limit] = -1
    infectious_ages[np.isnan(infectious_ages)] = -1
    return infectious_ages

class FerrettiInfectionDynamics(InfectionDynamics):
    '''
    Gamma distribution with shape 5.62 and scale 0.98, derived from known source-recipient pairs.

    Assumes that the incubation period distribution is also Gamma distributed with shape 5.807 and scale 0.948  

    Source: https://doi.org/10.1101/2020.09.04.20188516
    '''

    def __init__(self, shape: float = 5.62, scale: float = 0.98, upper_limit: int=21, label: str = None):
        """The Gamma distribution ends up being very difficult to calculate using Numba, due to
        the presence of a gamma function. Even if we only consider integer time steps, calculating factorials
        isn't something that can be easily done using Numba since it is typically found using recursion.

        If we make the assumption that the infectivity profile is the same for all cases, then we do not need to
        compute anything on the fly, instead we can actually just lookup the values from a dictionary.

        Args:
            shape (float): The shape parameter of the Gamma distribution. Defaults to 5.62.
            scale (float): The scale parameter of the Gamma distribution. Defaults to 0.98.
            upper_limit (int): The number of days for which lookup values are created. Defaults to 21. 
        """

        self.shape = shape
        self.scale = scale
        self.upper_limit = upper_limit
        self.precomputed_infection_dict = False

        super().__init__(label=label)


        self.precompute_dict()

        return

    def precompute_dict(self):
        """Precomputes values of the infectiousness profile and stores in a dictionary.s
        """

        # times at which we precompute values
        times = list(range(0, self.upper_limit + 1))

        # computing the value of the infectiousness profile at this time
        values = list(ss.gamma.pdf(x = times, a = 5.62, scale = 0.98))

        values = [float32(val) for val in values]
        
        # create dict and store
        self.lookup_dict = dict(zip(times, values))
        # if an individual has not been infected, they have a np.nan value
        self.lookup_dict[-1] = np.float32('nan')

        # record
        self.precomputed_infection_dict = True

    def compute_infectiousness(self, sim):
        """Performs a lookup operation and returns the infectiousness values

        Args:
            sim (cv.Sim): The inputted simulation object
        """
        
        t               = sim.t
        date_exposed    = sim.people.date_exposed 
        date_recovered  = sim.people.date_recovered
        date_dead       = sim.people.date_dead

        self.infectious_ages = get_infectious_ages(t, time_exposed=date_exposed, time_recovered=date_recovered, time_dead=date_dead, upper_limit=np.int32(self.upper_limit))

        return np.array([
            self.lookup_dict[age] for age in self.infectious_ages
        ])

    def plot(self):
        """Plots the infectiousness profile
        """

        x = np.linspace(0, 21, 100)
        plt.plot(x, ss.gamma.pdf(x = x, a = self.shape, scale = self.scale))
