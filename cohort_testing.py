import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import inspect
import datetime as dt
import covasim as cv
import covasim.utils as cvu
import covasim.defaults as cvd
import covasim.base as cvb
import covasim.parameters as cvpar
from covasim.interventions import process_daily_data, get_subtargets, get_quar_inds
from collections import defaultdict


class cohort_testing(cv.Intervention):
    '''
    Reimplement the class "test_prob" with sensitivity curves, bad swabbing and false positives.

    Args:
        symp_prob        (float)     : probability of testing a symptomatic (unquarantined) person
        asymp_prob       (float)     : probability of testing an asymptomatic (unquarantined) person (default: 0)
        symp_quar_prob   (float)     : probability of testing a symptomatic quarantined person (default: same as symp_prob)
        asymp_quar_prob  (float)     : probability of testing an asymptomatic quarantined person (default: same as asymp_prob)
        quar_policy      (str)       : policy for testing in quarantine: options are 'start' (default), 'end', 'both' (start and end), 'daily'; can also be a number or a function, see get_quar_inds()
        subtarget        (dict)      : subtarget intervention to people with particular indices  (see test_num() for details)
        ili_prev         (float/arr) : prevalence of influenza-like-illness symptoms in the population; can be float, array, or dataframe/series
        start_day        (int)       : day the intervention starts (default: 0, i.e. first day of the simulation)
        end_day          (int)       : day the intervention ends (default: no end)
        test_profile_key (str)       : key for test sensitivity curve
        kwargs           (dict)      : passed to Intervention()

    **Examples**::

        interv = cv.test_prob(symp_prob=0.1, asymp_prob=0.01) # Test 10% of symptomatics and 1% of asymptomatics
        interv = cv.test_prob(symp_quar_prob=0.4) # Test 40% of those in quarantine with symptoms
    '''
    def __init__(self, test_profile_key, cohort_size, anti_size, test_freq, test_prob, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object

        self.cohort_size = cohort_size
        self.anti_size = anti_size
        self.cohort_test_freq = test_freq
        self.remaining_test_prob = test_prob
        self.test_profile_key = test_profile_key
        return


    def initialize(self, sim):
        ''' Fix the dates '''

        pop_size = sim["pop_size"]
        in_cohort = int(self.cohort_size * pop_size)

        self.cohort = cvu.choose(pop_size, in_cohort)
        self.anti_testers = cvu.binomial_filter(self.anti_size, np.setdiff1d(np.arange(pop_size),self.cohort))
        self.less_reliable_testers = np.setdiff1d(np.arange(pop_size),np.union1d(self.cohort, self.anti_testers))

        self.initialized = True
        return


    def apply(self, sim):
        ''' Perform testing '''
        t = sim.t

        diag_inds = cvu.true(sim.people.diagnosed)
        test_probs = np.zeros(sim.n)
        symp_inds  = cvu.true(sim.people.symptomatic)

        if t % self.cohort_test_freq == 0:
            test_probs[self.cohort] = 1.

        test_probs[self.less_reliable_testers] = self.remaining_test_prob
        test_probs[diag_inds] = 0.
        test_probs[symp_inds] = 0.
        test_inds = cvu.true(cvu.binomial_arr(test_probs))

        sim.people.test_custom_profile(test_inds, test_sensitivity_profile_key = self.test_profile_key) # Actually test people
        sim.results['new_tests'][t] += int(len(test_inds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec
        sim.results[f'new_tests_{self.test_profile_key}'][t] += int(len(test_inds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

        return test_inds
