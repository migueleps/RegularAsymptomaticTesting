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
from contact_tracing_test import CT_no_quarantine
from cohort_testing import cohort_testing

ncpus = int(sys.argv[1])

uptakes_testing = np.arange(0,1.05,0.2)
uptakes_ct = np.arange(0,1.05,0.2)
uptakes_ili = np.arange(0,1.05,0.2)
test_specificity = 0.997
bad_swab_error = 0.1
#asympt_probs = [0.0, 1/7., 2/7., 3.5/7.]
#symp_prob = 0.25
trace_probs = {"h": 1., "s": 0.45, "w": 0.45, "c": 0.15}
trace_time = {"h": 0, "s": 1, "w": 1, "c": 2}
quar_period = 10
initial_infected = 100
runs_per_sim = 100
n_imports = 5
dynam_layer = {'c': True}
growth_rates = [0.025, 0.05, 0.1, 0.15, 0.225, 0.3]


def vaccinate_over40s_high(sim):
    over50s = cv.true(sim.people.age >= 40)
    under50s = cv.true(sim.people.age < 40)
    inds = sim.people.uid
    vals = np.ones(len(sim.people))
    vals[over50s] = 0.9
    vals[under50s] = 0.
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
pfizer_high = cv.vaccinate(vaccine = "pfizer", subtarget = vaccinate_under40s_high, days = 0)


lfa_curve = LFATestSensitivityCurve(lfa_pars_path = "data/lfa_pars.csv",
                                    test_specificity = test_specificity,
                                    swab_error_rate = bad_swab_error)


ct = {f"{uptake:.2f}":CT_no_quarantine(trace_probs = trace_probs,
                             trace_time = trace_time,
                             quar_period = quar_period,
                             test_profile_key = "lfa",
                             uptake = uptake) for uptake in uptakes_ct}


sympt_testing = {f"{uptake:.2f}":test_prob_with_curve(test_profile_key = "lfa",
                                                symp_prob = uptake,
                                                asymp_prob = 0.) for uptake in uptakes_testing}

ili = {f"{uptake:.2f}":ili_single_test(ili_prev = 0.01, adherence = uptake) for uptake in uptakes_ili}

ferretti_infectiousness = FerrettiInfectionDynamics()

iso_quar = isolation_quarantine_cost()

age_strata_outs = outputs_by_age()


sim_pars = dict(use_waning = True,
                location = "UK",
                rescale = False,
                pop_type = "matrix",
                pop_size = 100000,
                n_days = 180,
                test_sensitivity_profiles = {"lfa": lfa_curve},
                analyzers = [iso_quar, age_strata_outs],
                iso_period = quar_period,
                dynam_layer = dynam_layer,
                n_imports = n_imports,
                pop_infected = initial_infected)


interventions = {f"{k1}_{k2}_{k3}": [AZ_high, pfizer_high, ili[k1],sympt_testing[k2],ct[k3]] for k1 in ili for k2 in sympt_testing for k3 in ct}


def run_msim(sim_label, contact_scenario, growth_rate):
    contact_mat = contact_scenarios[contact_scenario]
    interv_comb = interventions[sim_label]
    calibrated_beta = calibration_dict[contact_scenario][growth_rate]
    sims = []
    for s in range(runs_per_sim):
        sim = cv.Sim(sim_pars,
                     rand_seed=s,
                     label = sim_label,
                     interventions = interv_comb,
                     contact_matrices = contact_mat,
                     infection_dynamics = ferretti_infectiousness,
                     beta = calibrated_beta)
        sims.append(sim)
    msim = cv.MultiSim(sims)
    msim.run(n_cpus=ncpus)
    msim.mean()
    msim.save(f"sympt_testing/LFD_SYMPT_INVESTIGATION_{contact_scenario}_{growth_rate}_{sim_label}.msim")

calibration_dict = {
    'September2020': {
        0.15: 0.672,
        0.225: 1.096,
        0.3: 2.034,
        0.025: 0.216,
        0.05: 0.294,
        0.1: 0.467},
    'Polymod': {
        0.025: 0.139,
        0.05: 0.181,
        0.1: 0.276,
        0.15: 0.387,
        0.225: 0.609,
        0.3: 1.049}
}

for contact_scenario in ["Polymod"]:
    for growth_rate in growth_rates:
        for sim_label in interventions:
            run_msim(sim_label, contact_scenario, growth_rate)
