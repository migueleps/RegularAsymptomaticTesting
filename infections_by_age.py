import numpy as np
import sciris as sc
import covasim as cv
import covasim.utils as cvu
import covasim.defaults as cvd


class outputs_by_age(cv.Analyzer):
    '''
    Analyzer that tracks the total amount of days people spend in quarantine and isolation to investigate the cost of increasing/decreasing these periods in relation to the infection rate and money spent by the government to accomodate people not going to work or to incentivize adherence to the quarantine and isolation policies.

    Args:
        kwargs (dict): passed to Analyzer()
    '''

    def __init__(self,  **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        return


    def initialize(self, sim):
        self.age_bins = [[0,5],[5,12],[12,18],[18,30],[30,40],[40,50],[50,60],[60,70],[70,120]]
        self.new_infections_by_age = np.full((len(self.age_bins),sim.npts),0,dtype=cvd.default_int)
        self.new_severe_by_age = np.full((len(self.age_bins),sim.npts),0,dtype=cvd.default_int)
        self.new_critical_by_age = np.full((len(self.age_bins),sim.npts),0,dtype=cvd.default_int)
        self.new_deaths_by_age = np.full((len(self.age_bins),sim.npts),0,dtype=cvd.default_int)
        self.initialized = True
        return


    def apply(self, sim):

        t = sim.t
        ages = sim.people.age

        for i, (lower, upper) in enumerate(self.age_bins):
            s_inds = sc.findinds((ages >= lower) * (ages < upper))
            self.new_infections_by_age[i,t] = len(cvu.true(sim.people.date_infectious[s_inds] == t))
            self.new_severe_by_age[i,t] = len(cvu.true(sim.people.date_severe[s_inds] == t))
            self.new_critical_by_age[i,t] = len(cvu.true(sim.people.date_critical[s_inds] == t))
            self.new_deaths_by_age[i,t] = len(cvu.true(sim.people.date_dead[s_inds] == t))

        return


    def get(self):

        return self.new_infections_by_age, self.new_severe_by_age, self.new_critical_by_age, self.new_deaths_by_age
