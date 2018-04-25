### Q1
# Part 1

Annual_Mortality_All     = 1800/100000
Annual_Mortality_Stroke  = 32.6/100000
Non_Stroke_Mort     = Annual_Mortality_All - Annual_Mortality_Stroke
print(Annual_Mortality_Stroke,Non_Stroke_Mort)

# Part 2

Annual_Rate_First_Stroke = 1500/100000
print(Annual_Rate_First_Stroke)

# Part 3

Stroke_Survive = Annual_Rate_First_Stroke * 0.9
Stroke_Death   = Annual_Rate_First_Stroke * 0.1
print(Stroke_Death, Stroke_Survive)

# Part 4

Annual_Recurrent = Stroke_Survive * 0.17 * 5
print(Annual_Recurrent)

# Part 5

Post_Stroke_Survive = Annual_Recurrent * 0.8
Post_Stroke_Death   = Annual_Recurrent * 0.2
print(Post_Stroke_Death, Post_Stroke_Survive)

# Part 6

Stroke_Duration = 7 / 365
Annual_Transition = 1 / Stroke_Duration
print(Annual_Transition)

###Q2
# Drug reduce 25% posst-stroke
Drug_Post_Stroke     = Post_Stroke_Survive * (1 - 0.25)
Drug_Non_Stroke_Mort = Non_Stroke_Mort * (1 + 0.05)
print(Drug_Post_Stroke, Drug_Non_Stroke_Mort)

###Q3/Q4
import scr.MarkovClasses as MarkovCls
import scr.RandomVariantGenerators as rndClasses
import scr.EconEvalClasses as EconCls
from enum import Enum

SIM_LENGTH = 15   # length of simulation (years) from 65 - 80 y/o
ALPHA = 0.05        # significance level for calculating confidence intervals
DELTA_T = 1         # years (length of time step, how frequently you look at the patient)
DISCOUNT = 0.03

No_Drug_Matrix=[
                        [None,Stroke_Survive,0,Stroke_Death,Non_Stroke_Mort],
                        [0,None,Post_Stroke_Survive,0,0],
                        [0,Post_Stroke_Survive,None,Post_Stroke_Survive,Non_Stroke_Mort],
                        [0,0,0,None,0],
                        [0,0,0,0,None]
    ]

Drug_Matrix = [
                        [None,Stroke_Survive,0,Stroke_Death,Non_Stroke_Mort],
                        [0,None,Post_Stroke_Survive,0,0],
                        [0,Drug_Post_Stroke,None,Drug_Post_Stroke,Drug_Non_Stroke_Mort],
                        [0,0,0,None,0],
                        [0,0,0,0,None]
    ]

Prob_Non_Drug_Matrix = MarkovCls.continuous_to_discrete(No_Drug_Matrix,delta_t)
Prob_Drug_Martix     = MarkovCls.continuous_to_discrete(Drug_Matrix,delta_t)


HEALTH_UTILITY = [
    1,  # well
    0.2,  # stroke
    0.9,  # post-stroke
    0  # dead
]

HEALTH_COST = [
    0,
    5000,  # stroke - one time cost
    200,  # post-stroke /year
    750, # when anticoagulation is used
    0
]

Anticoag_COST = 2000


from enum import Enum
import numpy as np
import scipy.stats as stat
import math as math
import scr.MarkovClasses as MarkovCls
import scr.RandomVariantGenerators as Random
import scr.FittingProbDist_MM as Est


class HealthStats(Enum):
    """ health states of patients with Stroke """
    Well = 0
    Stroke = 1
    Post_Stroke = 2
    Death_Stroke = 3
    Non_Stroke_Death = 4


class Drug(Enum):
    """ mono vs. combination therapy """
    No_Drug = 0
    Drug = 1


class _Parameters:

    def __init__(self, therapy):

        # selected therapy
        self._therapy = therapy

        # simulation time step
        self._delta_t = DELTA_T

        # calculate the adjusted discount rate
        self._adjDiscountRate = DISCOUNT * DELTA_T

        # initial health state
        self._initialHealthState = HealthStats.Well

        # annual treatment cost
        if self._therapy == Drug.No_Drug:
            self._annualTreatmentCost = 0
        else:
            self._annualTreatmentCost = Anticoag_COST

        # transition probability matrix of the selected therapy
        self._Drug_Matrix = []

        # treatment relative risk
        self._treatmentRR = 0

        # annual state costs and utilities
        self._annualStateCosts = []
        self._annualStateUtilities = []

    def get_initial_health_state(self):
        return self._initialHealthState

    def get_delta_t(self):
        return self._delta_t

    def get_adj_discount_rate(self):
        return self._adjDiscountRate

    def get_transition_prob(self, state):
        return self._No_Drug_Matrix[state.value]

    def get_annual_state_cost(self, state):
        if state == HealthStats.Death_Stroke or state == HealthStats.Non_Stroke_Death:
            return 0
        else:
            return self._annualStateCosts[state.value]

    def get_annual_state_utility(self, state):
        if state == HealthStats.Death_Stroke or state == HealthStats.Non_Stroke_Death:
            return 0
        else:
            return self._annualStateUtilities[state.value]

    def get_annual_treatment_cost(self):
        return self._annualTreatmentCost


class ParametersFixed(_Parameters):
    def __init__(self, therapy):

        # initialize the base class
        _Parameters.__init__(self, therapy)

        # calculate transition probabilities between hiv states
        self._no_drug_matrix = calculate_prob_matrix_no_drug()
        # add background mortality if needed
        if Data.ADD_BACKGROUND_MORT:
            add_non_stroke_death(self._no_drug_matrix)

        # update the transition probability matrix if combination therapy is being used
        if self._therapy == Drug.Drug:
            # treatment relative risk
            self._treatmentRR = Data.TREATMENT_RR
            # calculate transition probability matrix for the combination therapy
            self._prob_matrix = calculate_prob_matrix_combo(
                matrix_mono=self._no_drug_matrix, combo_rr=Data.TREATMENT_RR)

        # annual state costs and utilities
        self._annualStateCosts = Data.ANNUAL_STATE_COST
        self._annualStateUtilities = Data.ANNUAL_STATE_UTILITY


class ParametersProbabilistic(_Parameters):
    def __init__(self, seed, therapy):

        # initializing the base class
        _Parameters.__init__(self, therapy)

        self._rng = Random.RNG(seed)    # random number generator to sample from parameter distributions
        self._StorkeProbMatrixDrug = []  # list of dirichlet distributions for transition probabilities
        self._lnRelativeRiskDrug = None  # random variate generator for the natural log of the treatment relative risk
        self._annualStateCostDrug = []       # list of random variate generators for the annual cost of states
        self._annualStateUtilityDrug = []    # list of random variate generators for the annual utility of states

        # HIV transition probabilities
        j = 0
        for prob in Data.TRANS_MATRIX:
            self._StrokeProbMatrixDrug.append(Random.Dirichlet(prob[j:]))
            j += 1

        # treatment relative risk
        # find the mean and st_dev of the normal distribution assumed for ln(RR)
        sample_mean_lnRR = math.log(Data.TREATMENT_RR)
        sample_std_lnRR = \
            (math.log(Data.TREATMENT_RR_CI[1])-math.log(Data.TREATMENT_RR_CI[0]))/(2*stat.norm.ppf(1-0.05/2))
        self._lnRelativeRiskRVG = Random.Normal(loc=sample_mean_lnRR, scale=sample_std_lnRR)

        # annual state cost
        for cost in Data.ANNUAL_STATE_COST:
            # find shape and scale of the assumed gamma distribution
            estDic = Est.get_gamma_params(mean=cost, st_dev=cost/4)
            # append the distribution
            self._annualStateCostRVG.append(
                Random.Gamma(a=estDic["a"], loc=0, scale=estDic["scale"]))

        # annual state utility
        for utility in Data.ANNUAL_STATE_UTILITY:
            # find alpha and beta of the assumed beta distribution
            estDic = Est.get_beta_params(mean=utility, st_dev=utility/4)
            # append the distribution
            self._annualStateUtilityRVG.append(
                Random.Beta(a=estDic["a"], b=estDic["b"]))

        # resample parameters
        self.__resample()

    def __resample(self):

        # calculate transition probabilities
        # create an empty matrix populated with zeroes
        self._prob_matrix = []
        for s in HealthStats:
            self._prob_matrix.append([0] * len(HealthStats))

        # for all health states
        for s in HealthStats:
            # if the current state is death
            if s in [HealthStats.Death_Stroke, HealthStats.Non_Stroke_Death]:
                # the probability of staying in this state is 1
                self._No_Drug_Matrix[s.value][s.value] = 1
            else:
                # sample from the dirichlet distribution to find the transition probabilities between hiv states
                sample = self._StrokeProbMatrixDrug[s.value].sample(self._rng)
                for j in range(len(sample)):
                    self._prob_matrix[s.value][s.value+j] = sample[j]

        # add background mortality if needed
        if Data.ADD_BACKGROUND_MORT:
            add_non_stroke_death(self._No_Drug_Matrix)

        # update the transition probability matrix if combination therapy is being used
        if self._therapy == Drug.Drug:
            # treatment relative risk
            self._treatmentRR = math.exp(self._lnRelativeRiskDrug.sample(self._rng))
            # calculate transition probability matrix for the combination therapy
            self._prob_matrix = calculate_prob_matrix_combo(
                matrix_mono=self._prob_matrix, combo_rr=self._treatmentRR)

        # sample from gamma distributions that are assumed for annual state costs
        self._annualStateCosts = []
        for dist in self._annualStateCostDrug:
            self._annualStateCosts.append(dist.sample(self._rng))

        # sample from beta distributions that are assumed for annual state utilities
        self._annualStateUtilities = []
        for dist in self._annualStateUtilityDrug:
            self._annualStateUtilities.append(dist.sample(self._rng))


def calculate_prob_matrix_no_drug():
    """ :returns transition probability matrix for hiv states under mono therapy"""

    # create an empty matrix populated with zeroes
    prob_matrix = []
    for s in HealthStats:
        No_Drug_Matrix.append([0] * len(HealthStats))

    # for all health states
    for s in HealthStats:
        # if the current state is death
        if s in [HealthStats.Death_Stroke, HealthStats.Non_Stroke_Death]:
            # the probability of staying in this state is 1
            prob_matrix[s.value][s.value] = 1
        else:
            # calculate total counts of individuals
            sum_counts = sum(Data.TRANS_MATRIX[s.value])
            # calculate the transition probabilities out of this state
            for j in range(s.value, HealthStats.BACKGROUND_DEATH.value):
                prob_matrix[s.value][j] = Data.TRANS_MATRIX[s.value][j] / sum_counts

    return prob_matrix


def add_background_mortality(prob_matrix):

    # find the transition rate matrix
    rate_matrix = MarkovCls.discrete_to_continuous(prob_matrix, 1)
    # add mortality rates
    for s in HealthStats:
        if s not in [HealthStats.Death_Stroke, HealthStats.Non_Stroke_Death]:
            rate_matrix[s.value][HealthStats.Non_Stroke_Death.value] \
                = -np.log(1 - Data.ANNUAL_PROB_BACKGROUND_MORT)

    # convert back to transition probability matrix
    prob_matrix[:], p = MarkovCls.continuous_to_discrete(rate_matrix, Data.DELTA_T)
    # print('Upper bound on the probability of two transitions within delta_t:', p)


def calculate_prob_matrix_combo(matrix_no_drug, drug_rr):

    matrix_drug = []
    for l in matrix_no_drug:
        matrix_drug.append([0] * len(l))

    # populate the combo matrix
    # first non-diagonal elements
    for s in HealthStats:
        for next_s in range(s.value + 1, len(HealthStats)):
            matrix_drug[s.value][next_s] = drug_rr * matrix_no_drug[s.value][next_s]

    # diagonal elements are calculated to make sure the sum of each row is 1
    for s in HealthStats:
        if s not in [HealthStats.Death_Stroke, HealthStats.Non_Stroke_Death]:
            matrix_drug[s.value][s.value] = 1 - sum(matrix_drug[s.value][s.value + 1:])

    return matrix_drug
