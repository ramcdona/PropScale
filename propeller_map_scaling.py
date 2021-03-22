import numpy as np
import pandas as pd
from scipy.optimize import root
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapz
from scipy import interpolate
import pprint


# Helper functions definition
# ----------------------------------------------------------

# Prandtl approximation of Goldstein function
def G(x,NB,lam2):

    f = NB/2.0 * (1.0-x)*math.sqrt(1.0+lam2**2)/lam2
    F = 2.0/math.pi*math.acos(math.exp(-f))
    return F*x**2/(x**2+lam2**2)


# Calculation of the theodorsen kappa mass coefficient
def kTheod(NB,lam2):

    # Integrate 2G(x)xdx
    xintegr=np.linspace(start=0,stop=1, num=10,endpoint=True)
    yintegr=np.zeros_like(xintegr)

    for i,xi in enumerate(xintegr):
        yintegr[i] = 2.0*G(xi,NB,lam2)*xi

    return trapz(y=yintegr,x=xintegr)


# Calculation of the Epsilon (energy factor)
def eTheod(NB,lam2):

    dlam = 0.001
    ktheod2=kTheod(NB,lam2+dlam)

    if (lam2<dlam):
        ktheod1=kTheod(NB,lam2)
        e=kTheod(NB,lam2)+0.5*lam2*(ktheod2-ktheod1)/dlam
    else:
        ktheod1=kTheod(NB,lam2-dlam)

        e=kTheod(NB,lam2)+0.5*lam2*(ktheod2-ktheod1)/(2.*dlam)

    return e


def e_q_k(NB,lam2):
    return eTheod(NB,lam2)/kTheod(NB,lam2)


def func_Eta_i(w, J, CP, NB):
    lam0 = J/math.pi
    lam2 = lam0*(1.0+w)

    k2 = kTheod(NB,lam2)
    eqk = e_q_k(NB,lam2)

    KT1 = 2.0*k2*w*(1.0+w*(0.5+eqk))
    KP1 = 2.0*k2*w*(1.0+w)*(1.0+w*eqk)

    Eta_i = KT1/KP1

    KPgiven = CP * 8.0 / math.pi / J**3

    KP_error = KP1-KPgiven

    return KP_error, Eta_i


def func_KP_error(w,J,CP,NB):
    KP_error, _ = func_Eta_i(w, J, CP, NB)
    return KP_error


def Calc_gamma(J_i, Eta, Eta_i):
    Az = 1.0-3.0*(J_i/math.pi)**2.0+3.0*(J_i/math.pi)**3.0*math.atan(math.pi/J_i)
    Bz =3.0*J_i/(2.0*math.pi)*(1.0-(J_i/math.pi)**2.0*math.log((math.pi**2.0+J_i**2.0)/J_i**2.0))
    Z = Eta / Eta_i

    gamma = Bz/(1.0/(1.0-Z)-Az)
    return gamma


def Calc_CP_i(J_i,CP,gamma):

    Achi =1.0-3.0*(J_i/math.pi)**2.0+3.0*(J_i/math.pi)**3.0*math.atan(math.pi/J_i)
    Bchi = 3.0*J_i/(2.0*math.pi)*(1.0-(J_i/math.pi)**2.0*math.log((math.pi**2.0+J_i**2.0)/J_i**2.0))
    chi = 1.0/(1.0+gamma*Achi/Bchi)

    CP_i = CP*chi

    return CP_i


def Calc_Eta_i(J,CP,Eta,NB):

    # First calculate Eta_i using CP as input
    CP_input = CP

    # 2 iterations for Eta_i
    # the number of iterations can be increased if needed or
    # this can be replaced with a convergence criterion on (Cp_i-Cp_input)
    for iter in range(2):

        sol = root(fun=func_KP_error, x0=0.1, args=(J,CP_input,NB), method='hybr')
        w = sol.x

        KP_error, Eta_i = func_Eta_i(w, J, CP_input, NB)

        if Eta_i>0:
            J_i = J / Eta_i
        else:
            J_i = J

        # Use the estimated Eta_i to calculate the CP_i
        # and repeat the Eta_i calculation using CP_i
        gamma = Calc_gamma(J_i, max(Eta, 0), Eta_i)

        CP_i = Calc_CP_i(J_i,CP,gamma)

        CP_input = CP_i

    return Eta_i[0], J_i[0]



class ScalableMap():
    # Base (reference) propeller map definition
    base_map_path = "" # reference propeller map filepath
    base_NB = 3 # reference propeller number of blades
    base_AF = 100 # reference propeller activity factor

    # Scaled map parameters
    scaled_NB = 3 # Scaled propeller number of blades
    scaled_AF = 100 # Scaled propeller activity factor

    # Pitch definition relative radius (often 0.75 or 0.7)
    Rpitch = 0.75

    # Init function
    def __init__(self, base_map_path="", base_NB=3, base_AF=100):

        self.base_map_path = base_map_path
        self.base_NB = base_NB
        self.base_AF = base_AF

        # Begin with an unscaled map
        self.scaled_NB = base_NB
        self.scaled_AF = base_AF

    # Use this function to scale the propeller map
    def scale_map(self, scaled_NB, scaled_AF):

        self.scaled_NB = scaled_NB
        self.scaled_AF = scaled_AF


    # This function can be replaced if your interpolation works differently.
    # The map and the created interpolation function should be in the format (ETA, Pitch) = f(J, CP, Mach)
    def load_base_map(self):

        # Read a propeller map
        base_map = pd.read_csv(self.base_map_path)

        ETA_values = np.reshape(base_map.Eta.values, (len(base_map.Mach.unique()), len(base_map.J.unique()), len(base_map.CP.unique())))
        BETA_values = np.reshape(base_map.Beta.values, (len(base_map.Mach.unique()), len(base_map.J.unique()), len(base_map.CP.unique())))

        # Create interpolation function
        # Warning: using the following method extrapolation is not allowed and raises a ValueError
        self.ETA_interp_func = interpolate.RegularGridInterpolator((base_map.Mach.unique(), base_map.J.unique(), base_map.CP.unique()), ETA_values)
        self.BETA_interp_func = interpolate.RegularGridInterpolator((base_map.Mach.unique(), base_map.J.unique(), base_map.CP.unique()), BETA_values)

    # This function can be replaced if your interpolation works differently.
    # The map and the created interpolation function should be in the format (ETA, Pitch) = f(J, CP, Mach)
    def __interpolate_base_map(self, Mach, J, CP):

        ETA = self.ETA_interp_func([Mach, J, CP]).item()
        Beta = self.BETA_interp_func([Mach, J, CP]).item()

        return ETA, Beta


    def calculate_point_performance(self, Mach, J, Cp):

        output = {}

        # The J will stay the same we just need to convert the CP from scaled to base
        CPbase = Cp * (self.base_AF * self.base_NB)/(self.scaled_AF*self.scaled_NB)

        # Calculate the efficiency in the base map using CPbase
        Eta_CPbase, Beta_CPbase = self.__interpolate_base_map(Mach, J, CPbase)

        # Now calculate the ideal performance for the given J, CPbase
        Eta_i_CPbase, J_i_CPbase = Calc_Eta_i(J,CPbase,Eta_CPbase,self.base_NB)

        # Calculate the propeller losses
        Loss = Eta_i_CPbase - Eta_CPbase

        # Calculate the Alpha corresponding to the CPbase (Alpha considered constant between base and scaled)
        Alpha = Beta_CPbase - math.atan(J_i_CPbase/self.Rpitch/math.pi)*180./math.pi

        # Initial Eta guess
        EtaScaled_guess = 0.6

        # 10 iterations for EtaScaled_guess
        # Could be replaced with a convergence criterion
        for iiii in range(10):

            # First calculate the ideal performance for the given J, CP
            Eta_i, J_i = Calc_Eta_i(J, Cp, EtaScaled_guess, self.scaled_NB)

            # Calculate the efficiency of the scaled propeller
            EtaScaled = Eta_i - Loss

            # Calculate the Beta of the scaled propeller
            Beta_scaled = Alpha + math.atan(J_i/self.Rpitch/math.pi)*180./math.pi

            # Update the EtaScaled_guess
            EtaScaled_guess = 0.5*EtaScaled+0.5*EtaScaled_guess

        # Put results in a dictionary
        output["Eta_scaled"] = EtaScaled
        output["Beta_scaled"] = Beta_scaled
        output["Mach"] = Mach
        output["J"] = J
        output["CP"] = Cp

        return output


smap = ScalableMap(base_map_path="dummy_prop_map.csv", base_NB=3, base_AF=100)

smap.load_base_map()

results  = smap.calculate_point_performance(Mach=0.1, J=1.8, Cp=0.25)

print()
print("Base propeller map (NB=3, AF=100)")
print()
pprint.pprint(results)
print()

smap.scale_map(scaled_NB=4, scaled_AF=140)

results  = smap.calculate_point_performance(Mach=0.1, J=1.8, Cp=0.25)

print()
print("Scaled propeller map (NB=4, AF=140)")
print()
pprint.pprint(results)
print()

