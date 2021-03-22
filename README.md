# Propeller Scaling

This project contains a reference implementation in Python of the propeller scaling method described in:

Panagiotis Giannakakis, Ioannis Goulos, Panagiotis Laskaridis, Pericles Pilidis, and Anestis I. Kalfas,
*Novel Propeller Map Scaling Method*,
Journal of Propulsion and Power 2016 32:6, 1325-1332 [https://doi.org/10.2514/1.B35894](https://doi.org/10.2514/1.B35894)

This paper presents a novel method of scaling a baseline propeller map *&eta;_prop=f(J,CP,M)* in order
to obtain the performance of a propeller with different design characteristics. The developed method
employs a Goldstein/Theodorsen model to calculate the ideal efficiency and a simple approach to estimate
the propeller activity factor. The proposed scaling technique enables the use of a single propeller map
for propellers with different design flight conditions, diameters, number of blades, activity factors,
tip speeds, or power, provided that the blade sweep and airfoil distributions remain constant.
