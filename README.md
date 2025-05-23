# engine_sim_tools

Python toolkit for use with **AngeTheGreat Engine Simulator 2D**. This project provides tools to simulate engine port flow dynamics and calculate combustion geometry using reverse-engineered and empirical modeling.

## Features

- Head flow function generator (for both intake and exhaust)
- Interactive GUI with sliders to visualize and tune flow parameters
- Intelligent extrapolation and smoothing of flow curves
- Accurate reverse-engineering of Jeep 4.0 cylinder head and runner geometry
- Auto-generation of `.mr` files with flow functions, runner/collector dimensions, and more
- Conversion utilities (mm ↔ inch, cc ↔ CI)
- Runner geometry and harmonic resonance estimations
- Exhaust system tuning helpers (primary and collector sizing)
- Blow-by estimation
- Friction torque estimation based on displacement
- Supports `.mr` metadata extraction from embedded JSON configs
- Batch mode via CLI (`-f` option for file path)

## Script: `port_flow`

This module computes flow functions and geometric data based on engine configuration parameters read from `.mr` files. It visualizes results and exports them directly back into the `.mr`.

### Input

- `.mr` file with labeled values like:
  ```
  label bore(100)
  label stroke(120)
  label intake_valve_diameter(38)
  label exhaust_valve_diameter(32)
  ...
  ```
- JSON-style config under comment tag:
  ```c
  // {
  //   "flow_cfg": {
  //     "port_to_valve_area": 0.88,
  //     "primary_area_coeff": 1.1,
  //     ...
  //   }
  // }
  ```

### Output

- Graphical plot of flow functions
- Calculated properties (runner length, blow-by, valve geometry, volumes)
- Updates `.mr` with new:
  - `head` block
  - `intake_flow` & `exhaust_flow` functions
  - Flow rate tags and physical estimates

## Controls (Sliders in GUI)

| Parameter                     | Description                                                      |
|------------------------------|------------------------------------------------------------------|
| `flow rate mult`             | Multiplies flow rate (does not affect head flows)                |
| `port to valve area`         | Ratio of port CSA to valve CSA (default 0.88)                    |
| `valve to stem dia`          | Ratio of valve head diameter to stem (default 5)                 |
| `intake runner dia mult`     | Multiplier for intake runner diameter                            |
| `intake to exhaust runner ratio` | Ratio of intake runner vol to exhaust runner vol           |
| `primary area coeff`         | Ratio of primary pipe area to exhaust port area                  |
| `collector area coeff`       | Collector area divided by sum of primary areas                   |
| `smoothness`                 | Gaussian blur intensity for head flow curve                      |

## Examples

```bash
# Launch GUI and load engine
python _port_flow_v.py -f my_engine.mr
```

## Calculations Based On

- Jeep 4.0 flow data [PDF](https://cjclub.co.il/files/JEEP_4.0_PERFORMANCE_SPECS.pdf)
- Runner theory from [exx.se tech](https://www.exx.se/techinfo/runners/runners.html)
- Assumptions: CFM @ 28 inH₂O, harmonic tuning, sonic choke approximations

## Dependencies

```bash
pip install numpy matplotlib scipy comment_parser parse regex
```