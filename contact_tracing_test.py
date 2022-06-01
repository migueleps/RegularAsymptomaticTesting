import covasim as cv
import numpy as np
import sciris as sc
import covasim.utils as cvu
import covasim.defaults as cvd
from collections import defaultdict


class CT_no_quarantine(cv.Intervention):
    '''
    Contact tracing on double vaccinated according to government guidelines on 15/10/2021: traced contacts are asked to test with PCR

    Args:
        start_day (int): day the intervention starts (default is the first day)
        end_day (int): day the intervention ends
    '''

    def __init__(self, test_profile_key = "pcr",contact_limit = None, trace_probs = None, trace_time = None, start_day = 0, end_day = None, quar_period = None, uptake = 1., **kwargs):

        super().__init__(**kwargs)
        #self._store_args()
        #self.n_contacts_limit = contact_limit
        self.quar_period = quar_period
        self.trace_probs = trace_probs
        self.trace_time = trace_time
        self.start_day = start_day
        self.end_day = end_day
        self.contact_array_fp = {}
        self.contact_array_tp = {}
        self.test_profile_key = test_profile_key
        self.uptake = uptake


    def initialize(self, sim):

        super().initialize()
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        if self.trace_probs is None:
            self.trace_probs = 1.0
        if self.trace_time is None:
            self.trace_time = 0.0
        if self.quar_period is None:
            self.quar_period = sim['quar_period']
        if sc.isnumber(self.trace_probs):
            val = self.trace_probs
            self.trace_probs = {k:val for k in sim.people.layer_keys()}
        if sc.isnumber(self.trace_time):
            val = self.trace_time
            self.trace_time = {k:val for k in sim.people.layer_keys()}
        self.initialized = True


    def apply(self, sim):
        '''
        Trace and notify contacts

        Tracing involves three steps that can independently be overloaded or extended
        by derived classes

        - Select which confirmed cases get interviewed by contact tracers
        - Identify the contacts of the confirmed case
        - Notify those contacts that they have been exposed and need to take some action
        '''

        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        trace_inds = self.select_cases(sim)
        self.identify_contacts(sim, trace_inds)
        timestep_inds_tp = np.array(self.contact_array_tp.pop(sim.t,[]), dtype=cvd.default_int)
        timestep_inds_fp = np.array(self.contact_array_fp.pop(sim.t,[]), dtype=cvd.default_int)

        timestep_inds = np.union1d(timestep_inds_tp, timestep_inds_fp)
        wrong_contacts = np.setdiff1d(timestep_inds_fp, timestep_inds_tp) #inds in FP contacts and not in TP contacts are wrong contacts

        self.save_wrong_contacts(sim, wrong_contacts)
        sim.people.wrong_quarantines[timestep_inds_tp] = False

        self.test_contacts(sim, timestep_inds)


    def select_cases(self, sim):
        '''
        Return people that were diagnosed this time step
        '''
        inds = cvu.true(sim.people.date_diagnosed == sim.t) # Diagnosed this time step, time to trace
        return inds


    def identify_contacts(self, sim, trace_inds):
        '''
        Return contacts to notify by trace time

        In the base class, the trace time is the same per-layer, but derived classes might
        provide different functionality e.g. sampling the trace time from a distribution. The
        return value of this method is a dict keyed by trace time so that the `Person` object
        can be easily updated in `contact_tracing.notify_contacts`

        Args:
            sim: Simulation object
            trace_inds: Indices of people to trace

        Returns: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''

        if not len(trace_inds):
            return

        true_pos_inds = np.intersect1d(trace_inds, cvu.true(sim.people.infectious))
        false_pos_inds = np.intersect1d(trace_inds, cvu.false(sim.people.infectious))


        for lkey, this_trace_prob in self.trace_probs.items():

            if this_trace_prob == 0:
                continue

            traceable_inds_tp = sim.people.contacts[lkey].find_contacts(true_pos_inds)
            traceable_inds_fp = sim.people.contacts[lkey].find_contacts(false_pos_inds)

            if len(traceable_inds_tp):
                contacts = cvu.binomial_filter(this_trace_prob, traceable_inds_tp)# Filter the indices according to the probability of being able to trace this layer
                existing_contacts = self.contact_array_tp.get(sim.t + self.trace_time[lkey], [])
                self.contact_array_tp[sim.t + self.trace_time[lkey]] = np.unique(np.concatenate([contacts, existing_contacts]))

            if len(traceable_inds_fp):
                contacts = cvu.binomial_filter(this_trace_prob, traceable_inds_fp)# Filter the indices according to the probability of being able to trace this layer
                existing_contacts = self.contact_array_fp.get(sim.t + self.trace_time[lkey], [])
                self.contact_array_fp[sim.t + self.trace_time[lkey]] = np.unique(np.concatenate([contacts, existing_contacts]))


    def test_contacts(self, sim, contact_inds):
        '''

        Args:
            sim: Simulation object
            contacts: {trace_time: np.array(inds)} dictionary storing which people to notify
        '''
        is_dead = cvu.true(sim.people.dead) # Find people who are not alive
        contact_inds = np.setdiff1d(contact_inds, is_dead) # Do not notify contacts who are dead
        #print("CONTACT INDS")
        #print(contact_inds)
        sim.people.known_contact[contact_inds] = True
        sim.people.date_known_contact[contact_inds] = np.fmin(sim.people.date_known_contact[contact_inds], sim.t)

        inds_who_test = cvu.binomial_filter(self.uptake, contact_inds)

        sim.people.test_custom_profile(inds_who_test, test_sensitivity_profile_key = self.test_profile_key)
        sim.results['new_tests'][sim.t] += int(len(inds_who_test)*sim['pop_scale']/sim.rescale_vec[sim.t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec
        sim.results[f'new_tests_{self.test_profile_key}'][sim.t] += int(len(inds_who_test)*sim['pop_scale']/sim.rescale_vec[sim.t]) # If we're using dynamic scaling, we have to scale by pop_scale, not rescale_vec
        return

    def save_wrong_contacts(self, sim, contacts):
        sim.people.flows["new_wrong_quarantines"] += len(contacts)
        sim.people.wrong_quarantines[contacts] = True
