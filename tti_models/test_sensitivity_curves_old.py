'''
Contains code that samples lateral flow testing parameters from posterior distributions,
and initialises some test sensitivity profiles
'''
import pandas as pd
import numpy as np
import numpy.random as npr
import covasim.utils as cvu
import covasim.defaults as cvd
from functools import partial
from covasim.test_sensitivity_profile import TestSensitivityProfile


# TODO: can this be vectorised? Does it need to be vectorised? Probably ought to vectorise
# TODO: would be nice to add a plot function for the various curves

def step_function(infectious_age, breakpoint_par):
    if (infectious_age > breakpoint_par):
        return 1
    else:
        return 0

def inv_logit(x):
    return np.exp(x) / (np.exp(x) + 1)

# returns a test sensitivity curve from a piecewise logistic regression
def positivity_curve(infectious_age, breakpoint_par, intercept, slope_regression_1, slope_regression_2):
    time_relative_to_breakpoint = (infectious_age - breakpoint_par)
    coefficient = intercept + slope_regression_1 * time_relative_to_breakpoint + slope_regression_1 * slope_regression_2 * time_relative_to_breakpoint * step_function(
        infectious_age, breakpoint_par)
    return inv_logit(coefficient)

class PCRTestSensitivityCurve(TestSensitivityProfile):

    def __init__(self, pcr_pars_path, loss_prob = 0.0, test_specificity = 1., swab_error_rate = 0.):

        self.loss_prob = loss_prob
        self.test_delay = dict(dist="poisson", par1 = 1.2)
        self.test_specificity = test_specificity
        self.swab_error_rate = swab_error_rate

        # get parameters from the PCR posterior distribution
        self.pcr_pars = pd.read_csv(pcr_pars_path, index_col=0)
        pcr_pars = self.get_pars()

        # create a pcr sensitivity curve
        self.pcr_prob_positive = partial(
            positivity_curve,
            breakpoint_par=pcr_pars.breakpoint_par,
            intercept=pcr_pars.intercept,
            slope_regression_1=pcr_pars.slope_regression_1,
            slope_regression_2=pcr_pars.slope_regression_2
        )

    # sample from the pcr par posterior
    def get_pars(self):
        index = npr.choice(list(range(1, 4000)))
        return self.pcr_pars.loc[index, :]

    def test(self, people, inds):

        # extract useful parameters
        t = people.t

        # compute the infectious ages of infected individuals
        inds = np.unique(inds)
        people.tested[inds]      = True
        people.date_tested[inds] = t # Only keep the last time they tested
        test_delays = cvu.sample(**self.test_delay, size=len(people))
        people.date_test_result[inds] = t + test_delays[inds]

        bad_swab = cvu.n_binomial(self.swab_error_rate, len(inds))
        bad_swab_negatives = inds[bad_swab]

        is_infectious            = cvu.itruei(people.infectious, inds) # only care about infectious individuals
        infectious_age_vec       = t - people.date_infectious[is_infectious]

        # compute the probability of testing positive, for each individual being tested
        prob_positive = [
            self.pcr_prob_positive(age)
            for age
            in infectious_age_vec
        ]

        pos_test      = cvu.binomial_arr(prob_arr=prob_positive)
        is_inf_pos    = is_infectious[pos_test]

        is_well = cvu.ifalsei(people.infectious, inds)
        pos_test = cvu.n_binomial(1-self.test_specificity, len(is_well))
        is_well_pos = is_well[pos_test]

        all_positives = np.unique(np.concatenate([is_inf_pos,is_well_pos]))
        not_lost      = cvu.n_binomial(1.0-self.loss_prob, len(all_positives))
        final_inds    = all_positives[not_lost]
        final_inds    = final_inds[~np.in1d(final_inds,bad_swab_negatives)]

        people.date_diagnosed[final_inds]   = t + test_delays[final_inds]
        people.date_pos_test[final_inds]    = t

        people.schedule_isolation(final_inds, start_date = t + test_delays[final_inds])


class LFATestSensitivityCurve(PCRTestSensitivityCurve):

    def __init__(self, lfa_pars_path, loss_prob = 0.0, test_specificity = 1., swab_error_rate = 0.):

        self.loss_prob = loss_prob
        self.test_specificity = test_specificity
        self.swab_error_rate = swab_error_rate

        # get parameters from the LFA posterior distribution
        self.lfa_pars = pd.read_csv(lfa_pars_path, index_col=0)
        lfa_pars = self.get_pars()

        # create a LFA sensitivity curve based on the posterior sample
        self.lfa_prob_positive = partial(
            positivity_curve,
            breakpoint_par=lfa_pars.breakpoint_par,
            intercept=lfa_pars.intercept,
            slope_regression_1=lfa_pars.slope_regression_1,
            slope_regression_2=lfa_pars.slope_regression_2
        )

    def get_pars(self):

        index = npr.choice(list(range(1, 4000)))
        return self.lfa_pars.loc[index, :]

    def test(self, people, inds):

        # extract useful parameters
        t = people.t

        # compute the infectious ages of infected individuals
        inds = np.unique(inds)
        people.tested[inds]      = True
        people.date_tested[inds] = t # Only keep the last time they tested
        people.date_test_result[inds] = t

        bad_swab = cvu.n_binomial(self.swab_error_rate, len(inds))
        bad_swab_negatives = inds[bad_swab]

        is_infectious            = cvu.itruei(people.infectious, inds) # only care about infectious individuals
        infectious_age_vec       = t - people.date_infectious[is_infectious]

        # compute the probability of testing positive, for each individual being tested
        prob_positive = [
            self.lfa_prob_positive(age)
            for age
            in infectious_age_vec
        ]

        pos_test      = cvu.binomial_arr(prob_arr=prob_positive)
        is_inf_pos    = is_infectious[pos_test]

        is_well = cvu.ifalsei(people.infectious, inds)
        pos_test = cvu.n_binomial(1-self.test_specificity, len(is_well))
        is_well_pos = is_well[pos_test]

        all_positives = np.unique(np.concatenate([is_inf_pos,is_well_pos]))
        not_lost      = cvu.n_binomial(1.0-self.loss_prob, len(all_positives))
        final_inds    = all_positives[not_lost]
        final_inds = final_inds[~np.in1d(final_inds,bad_swab_negatives)]

        people.date_diagnosed[final_inds]   = t
        people.date_pos_test[final_inds]    = t

        people.schedule_isolation(final_inds, start_date = t)
