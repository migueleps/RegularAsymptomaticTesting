import covasim as cv
import numpy as np
import sciris as sc
import covasim.utils as cvu
import covasim.defaults as cvd
from collections import defaultdict


class ili_single_test(cv.Intervention):
    '''
    An intervention to generate ILI symptoms and isolate anyone with symptoms (COVID or ILI). Gets released after testing negative once.

    Args:
        ili_prev (float): percentage of the population to infect with ILI per day.
        adherence (float): percentage of the population who actually adheres to the policy
    '''

    def __init__(self, ili_prev, adherence, **kwargs):

        super().__init__(**kwargs)
        #self._store_args()
        self.ili_prev = ili_prev
        self.adherence = adherence
        self.ili2rec = dict(dist='lognormal_int', par1=6.0,  par2=2.0) #average less 2 days than covid asymp and mild symp


    def initialize(self, sim):

        self.initialized = True



    def apply(self, sim):
        '''
        Isolate people with symptoms
        '''

        t = sim.t
        pop_size = sim["pop_size"]

        # Infect new people with ILI
        n_ili = int(self.ili_prev * pop_size)
        new_ili_inds = cvu.choose(pop_size, n_ili)
        sim.people.date_ili_recover[new_ili_inds] = cvu.sample(**self.ili2rec, size=n_ili) + t
        sim.people.ili[new_ili_inds] = True
        sim.people.date_ili[new_ili_inds] = t

        # Choose the people to isolate because of symptoms in this timestep
        symp_inds = cvu.true(sim.people.date_symptomatic == t)
        isolate_inds = np.union1d(new_ili_inds,symp_inds)

        isolate_inds = np.setdiff1d(isolate_inds, cvu.true(sim.people.quarantined))
        isolate_inds = isolate_inds[cvu.choose(len(isolate_inds),self.adherence * len(isolate_inds))]

        sim.people.schedule_isolation(isolate_inds, start_date = t)

        # Check if the people who recover from ILI
        ili_rec_inds = cvu.true(sim.people.date_ili_recover == t)
        sim.people.ili[ili_rec_inds] = False


        # Check the people with ILI who tested negative so they can be released from isolation

        # get the people who have ILI but were not infected in this timestep and are in isolation
        ili_inds = np.setdiff1d(cvu.true(sim.people.ili),new_ili_inds)
        ili_iso = np.intersect1d(ili_inds, cvu.true(sim.people.diagnosed))

        # get the people who tested after being infected with ILI (date of ILI lesser than date of test)
        tested_after_ili = cvu.itrue(sim.people.date_ili[ili_iso] < sim.people.date_test_result[ili_iso], ili_iso)

        # from the people who tested after being infected with ILI, get the ones who tested negative (those whose date of the last positive test is not after being infected with ILI)
        tested_negative = cvu.ifalse(sim.people.date_ili[tested_after_ili] < sim.people.date_diagnosed[tested_after_ili], tested_after_ili)

        # mark them to be released from isolation (check_diag is called after all the interventions, so if the date of diagnosis end is this time step, they will be marked as not isolating)
        sim.people.date_end_diagnosis[tested_negative] = t

        # Need to do the same for symptomatic people, if they tested negative after becoming symptomatic, they should not be isolating
        symp_inds = cvu.true(sim.people.symptomatic)
        symp_iso = np.intersect1d(symp_inds, cvu.true(sim.people.diagnosed))
        # get those who tested after developing symptoms
        tested_after_symp = cvu.itrue(sim.people.date_symptomatic[symp_iso] < sim.people.date_test_result[symp_iso], symp_iso)
        # of those, get the ones who tested negative
        tested_negative = cvu.ifalse(sim.people.date_symptomatic[tested_after_symp] < sim.people.date_diagnosed[tested_after_symp], tested_after_symp)
        sim.people.date_end_diagnosis[tested_negative] = t
