# non-linear progression generator
import numpy as np

num_values = 16

start_val = 11.0
end_val = 110.0
steepness_factor = 0.85

x_values = np.linspace(0, 1, num_values)
values = start_val + (end_val - start_val) * np.power(x_values, steepness_factor)
step = 5.0

res = f"        function({step}) // steepness {steepness_factor}\n"
turbulence = 0
for v in values:
    res += f"            .add_sample({turbulence:.1f}, {v:.1f})\n"
    turbulence += step
res += "    ;\n}\n"

print(res)