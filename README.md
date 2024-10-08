# engine_sim_tools

python toolkit for AngeTheGreat Engine Simulator 2D

## port_flow

This script is designed to generate head flow functions based on basic engine parameters. By reading key data such as bore, stroke, valve diameters, and lifts from a file, the tool computes both intake and exhaust flow functions. It also generates performance-related parameters such as cross-sectional areas, runner flows, blowby, and more.

Head flow calculations are based on a reverse-engineered head flow report from a Jeep 4.0 engine.

The script features a simple GUI to interactively control engine parameters and display results.

An `.mr` file should have specific labels and sections for this script to work. Refer to the `reference_engine.mr` file for more details.

### Controls

Some parameters are immutable (bore, stroke, number of cylinders) and are parsed from labels. Other parameters, like ratios and coefficients, are mutable and are parsed from JSON comments. Mutable parameters are controlled via sliders.

+ `flow rate mult` - Restricts or multiplies intake and exhaust flow rates; does not affect head flows. A value of 0 is fully restricted, 2 is the maximum multiplier, and the default is 1.
+ `port to valve area` - The ratio of _port_ / _valve cross-sectional area_, ranges from 0.75 to 1.
+ `valve to stem dia` - The ratio of valve _head diameter_ / _stem diameter_, ranges from 3 to 7.
+ `intake runner dia mult` - Adjusts the diameter of the intake runner, ranging from 0.8 to 1.2, where 1 leaves it unchanged.
+ `intake to exhaust runner ratio` - The ratio of _intake runner volume_ / _exhaust runner volume_, ranges from 1 to 5.
+ `primary area coeff` - The ratio of _primary cross-sectional area_ / _port cross-sectional area_, mainly affects sound.
+ `collector area coeff` - The ratio of _exhaust collector area_ / _sum of all primary cross-sectional areas_, essentially controlling the size of the exhaust pipe.
+ `smoothness` - Adjusts the smoothness of the head flow function.