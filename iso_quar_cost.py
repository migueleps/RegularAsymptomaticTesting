import numpy as np
import sciris as sc
import covasim as cv
import covasim.utils as cvu
import covasim.defaults as cvd


class isolation_quarantine_cost(cv.Analyzer):
    '''
    Analyzer that tracks the total amount of days people spend in quarantine and isolation to investigate the cost of increasing/decreasing these periods in relation to the infection rate and money spent by the government to accomodate people not going to work or to incentivize adherence to the quarantine and isolation policies.

    Args:
        kwargs (dict): passed to Analyzer()
    '''

    def __init__(self,  **kwargs):
        super().__init__(**kwargs) # Initialize the Analyzer object
        return


    def initialize(self, sim):
        self.quarantine_ttr_days = np.full(sim.people.pars["pop_size"], 0, dtype = cvd.default_int)
        self.quarantine_ttr_days_FP = np.full(sim.people.pars["pop_size"], 0, dtype = cvd.default_int)
        self.isolation_days = np.full(sim.people.pars["pop_size"], 0, dtype = cvd.default_int)
        self.isolation_days_FP = np.full(sim.people.pars["pop_size"], 0, dtype = cvd.default_int)
        self.isolating_from_FP = np.full(sim.npts, 0, dtype = cvd.default_int)
        self.isolating_and_negative = np.full(sim.npts, 0, dtype=cvd.default_int)
        self.ages = sim.people.age
        self.initialized = True
        return


    def apply(self, sim):

        isolation_inds = cvu.true(sim.people.diagnosed)
        infectious_inds = cvu.true(sim.people.infectious)
        true_positive_inds = np.intersect1d(isolation_inds, infectious_inds) # people who are isolating and are infectious
        false_positive_inds = np.setdiff1d(isolation_inds, true_positive_inds) # take away the people who are infectious from the people who are isolating

        self.isolating_and_negative[sim.t] = len(false_positive_inds)

        # People may isolate from ILI symptoms. If someone is isolating from ILI, they either:
        # - Haven't tested yet (they won't be in infectious inds and they shouldn't be in false positive inds, so we need to remove them)
        # - Tested positive and are covid positive (they will be in infectious inds so will be removed from false positive inds)
        # - Tested positive and are covid negative (they will be in isolation inds and not in infectious inds so they are automatically covered)
        # - Tested negative and are covid positive (they will not be marked as ILI)
        # - Tested negative and are covid negative (they will not be marked as ILI)
        ili_inds = cvu.true(sim.people.ili)
        ili_inds_no_test = cvu.itrue(sim.people.date_ili[ili_inds] >= sim.people.date_test_result[ili_inds], ili_inds)

        false_positive_inds = np.setdiff1d(false_positive_inds, ili_inds_no_test)

        quarantine_inds = cvu.true(sim.people.quarantined)
        ttr_inds = cvu.true(sim.people.ttr)
        q_inds = np.union1d(quarantine_inds, ttr_inds)
        q_inds = np.setdiff1d(q_inds,isolation_inds)
        q_inds_FP = np.intersect1d(q_inds,cvu.true(sim.people.wrong_quarantines))

        self.quarantine_ttr_days[q_inds] += 1
        self.quarantine_ttr_days_FP[q_inds_FP] += 1


        self.isolation_days[isolation_inds] += 1
        self.isolation_days_FP[false_positive_inds] += 1
        self.isolating_from_FP[sim.t] = len(false_positive_inds)

        return


    def get(self):

        return self.quarantine_ttr_days, self.quarantine_ttr_days_FP, self.isolation_days, self.isolation_days_FP, self.isolating_fom_FP, self.ages, self.isolating_and_negative
