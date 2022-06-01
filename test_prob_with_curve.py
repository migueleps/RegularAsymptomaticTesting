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


class test_prob_with_curve(cv.Intervention):
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
    def __init__(self, test_profile_key, symp_prob, asymp_prob=0.0, symp_quar_prob=None, asymp_quar_prob=None, quar_policy=None, subtarget=None, start_day=0, end_day=None, swab_delay=None, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self.symp_prob        = symp_prob
        self.asymp_prob       = asymp_prob
        self.symp_quar_prob   = symp_quar_prob  if  symp_quar_prob is not None else  symp_prob
        self.asymp_quar_prob  = asymp_quar_prob if asymp_quar_prob is not None else asymp_prob
        self.quar_policy      = quar_policy if quar_policy else 'start'
        self.subtarget        = subtarget
        self.start_day        = start_day
        self.end_day          = end_day
        self.pdf              = cvu.get_pdf(**sc.mergedicts(swab_delay)) # If provided, get the distribution's pdf -- this returns an empty dict if None is supplied
        self.test_profile_key = test_profile_key
        return


    def initialize(self, sim):
        ''' Fix the dates '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        self.initialized = True
        return


    def apply(self, sim):
        ''' Perform testing '''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Find probablity for symptomatics to be tested
        symp_inds  = cvu.true(sim.people.symptomatic)
        symp_prob = self.symp_prob
        if self.pdf:
            symp_time = cvd.default_int(t - sim.people.date_symptomatic[symp_inds]) # Find time since symptom onset
            inv_count = (np.bincount(symp_time)/len(symp_time)) # Find how many people have had symptoms of a set time and invert
            count = np.nan * np.ones(inv_count.shape)
            count[inv_count != 0] = 1/inv_count[inv_count != 0]
            symp_prob = np.ones(len(symp_time))
            inds = 1 > (symp_time*self.symp_prob)
            symp_prob[inds] = self.symp_prob/(1-symp_time[inds]*self.symp_prob)
            symp_prob = self.pdf.pdf(symp_time) * symp_prob * count[symp_time]

        # Define symptomatics, accounting for ILI prevalence
        pop_size = sim['pop_size']
        ili_inds = cvu.true(sim.people.ili)

        # Define asymptomatics: those who neither have COVID symptoms nor ILI symptoms
        asymp_inds = np.setdiff1d(np.setdiff1d(np.arange(pop_size), symp_inds), ili_inds)

        # Handle quarantine and other testing criteria
        quar_test_inds = get_quar_inds(self.quar_policy, sim)
        symp_quar_inds  = np.intersect1d(quar_test_inds, symp_inds)
        asymp_quar_inds = np.intersect1d(quar_test_inds, asymp_inds)
        diag_inds       = cvu.true(sim.people.diagnosed)

        symp_and_diag = np.intersect1d(symp_inds, diag_inds)
        diag_without_test = cvu.itrue(sim.people.date_symptomatic[symp_and_diag] > sim.people.date_tested[symp_and_diag], symp_and_diag)

        diag_inds = np.setdiff1d(diag_inds, ili_inds)
        diag_inds = np.setdiff1d(diag_inds, diag_without_test)

        # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
        test_probs = np.zeros(sim.n) # Begin by assigning equal testing probability to everyone
        test_probs[symp_inds]       = symp_prob            # People with symptoms (true positive)
        test_probs[ili_inds]        = symp_prob            # People with symptoms (false positive)
        test_probs[asymp_inds]      = self.asymp_prob      # People without symptoms
        test_probs[symp_quar_inds]  = self.symp_quar_prob  # People with symptoms in quarantine
        test_probs[asymp_quar_inds] = self.asymp_quar_prob # People without symptoms in quarantine
        if self.subtarget is not None:
            subtarget_inds, subtarget_vals = get_subtargets(self.subtarget, sim)
            test_probs[subtarget_inds] = subtarget_vals # People being explicitly subtargeted
        test_probs[diag_inds] = 0.0 # People who are diagnosed don't test
        test_inds = cvu.true(cvu.binomial_arr(test_probs)) # Finally, calculate who actually tests

        # Actually test people
        sim.people.test_custom_profile(test_inds, test_sensitivity_profile_key = self.test_profile_key) # Actually test people
        sim.results['new_tests'][t] += int(len(test_inds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec
        sim.results[f'new_tests_{self.test_profile_key}'][t] += int(len(test_inds)*sim['pop_scale']/sim.rescale_vec[t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec

        return test_inds
