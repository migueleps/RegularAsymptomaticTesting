import covasim as cv
import numpy as np
from daily_contact_testing import daily_contact_testing
from iso_quar_cost import isolation_quarantine_cost
from test_prob_with_curve import test_prob_with_curve
from tti_models.test_sensitivity_curves import PCRTestSensitivityCurve, LFATestSensitivityCurve
from contact_tracing_limited import limited_contact_tracing
from ili_DT import ili_daily_test
from ili_single_test import ili_single_test
from infections_by_age import outputs_by_age
from copy import copy
from tti_models.infectiousness_profiles import FerrettiInfectionDynamics
from contact_matrices import contact_scenarios
import sys

test_specificity = 0.997
bad_swab_error = 0.1
asympt_probs = [0.001, 0.01, 0.1]
symp_prob = 0.25
trace_probs = {"h": 1., "s": 0.45, "w": 0.45, "c": 0.15}
trace_time = {"h": 0, "s": 1, "w": 1, "c": 2}
quar_period = 10
initial_infected = 100
runs_per_sim = 100
n_imports = 5
dynam_layer = {'c': True}


def vaccinate_over40s_high(sim):
    over50s = cv.true(sim.people.age >= 40)
    under50s = cv.true(sim.people.age < 40)
    inds = sim.people.uid
    vals = np.ones(len(sim.people))
    vals[over50s] = 0.9
    vals[under50s] = 0.
    output = dict(inds = inds, vals = vals)
    return output

def vaccinate_over40s_low(sim):
    over50s = cv.true(sim.people.age >= 40)
    under50s = cv.true(sim.people.age < 40)
    inds = sim.people.uid
    vals = np.ones(len(sim.people))
    vals[over50s] = 0.75
    vals[under50s] = 0.
    output = dict(inds = inds, vals = vals)
    return output

def vaccinate_under40s_low(sim):
    over50s = cv.true(sim.people.age >= 40)
    under50s = cv.true(sim.people.age < 40)
    inds = sim.people.uid
    vals = np.ones(len(sim.people))
    vals[over50s] = 0.
    vals[under50s] = 0.2
    output = dict(inds = inds, vals = vals)
    return output

def vaccinate_under40s_high(sim):
    over50s = cv.true(sim.people.age >= 40)
    under50s = cv.true(sim.people.age < 40)
    inds = sim.people.uid
    vals = np.ones(len(sim.people))
    vals[over50s] = 0.
    vals[under50s] = 0.45
    output = dict(inds = inds, vals = vals)
    return output


AZ_high = cv.vaccinate(vaccine = "az", subtarget = vaccinate_over40s_high, days = 0)
AZ_low = cv.vaccinate(vaccine = "az", subtarget = vaccinate_over40s_low, days = 0)
pfizer_high = cv.vaccinate(vaccine = "pfizer", subtarget = vaccinate_under40s_high, days = 0)
pfizer_low = cv.vaccinate(vaccine = "pfizer", subtarget = vaccinate_under40s_low, days = 0)

lfa_curve = LFATestSensitivityCurve(lfa_pars_path = "data/lfa_pars.csv",
                                    test_specificity = test_specificity,
                                    swab_error_rate = bad_swab_error)

pcr_curve = PCRTestSensitivityCurve(pcr_pars_path = "data/pcr_pars.csv",
                                    test_specificity = test_specificity,
                                    swab_error_rate = bad_swab_error)

testing_intervs = {a: test_prob_with_curve(test_profile_key = "lfa",
                                           symp_prob = 0.,
                                           asymp_prob = a) for a in asympt_probs}

lfd_symptomatic_testing = test_prob_with_curve(test_profile_key = "lfa",
                                                symp_prob = symp_prob,
                                                asymp_prob = 0.)

pcr_symptomatic_testing = test_prob_with_curve(test_profile_key = "pcr",
                                                symp_prob = symp_prob,
                                                asymp_prob = 0.)


ct = limited_contact_tracing(trace_probs = trace_probs,
                             trace_time = trace_time,
                             quar_period = quar_period)

dct = daily_contact_testing(test_profile_key = "lfa",
                            n_days = quar_period,
                            trace_probs = trace_probs,
                            trace_time = trace_time)

ili_ST_iso = ili_single_test(ili_prev = 0.01, adherence = 0.4)
ili_DT = ili_daily_test(ili_prev = 0.01, adherence = 0.6)

ferretti_infectiousness = FerrettiInfectionDynamics()

iso_quar = isolation_quarantine_cost()

age_strata_outs = outputs_by_age()


sim_pars = dict(use_waning = True,
                location = "UK",
                rescale = False,
                pop_type = "matrix",
                beta = 0.73,
                pop_size = 100000,
                n_days = 180,
                test_sensitivity_profiles = {"lfa": lfa_curve, "pcr": pcr_curve},
                analyzers = [iso_quar, age_strata_outs],
                iso_period = quar_period,
                dynam_layer = dynam_layer,
                n_imports = n_imports,
                pop_infected = initial_infected)
