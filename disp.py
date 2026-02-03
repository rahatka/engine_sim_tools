from fractions import Fraction
import math

def parse_fraction(s):
    s = s.strip()
    if ' ' in s:
        whole, frac = s.split()
        return float(Fraction(whole)) + float(Fraction(frac))
    return float(Fraction(s))

def calculate_displacement(input_str):
    parts = input_str.split(',')
    if len(parts) != 3:
        raise ValueError("Expected format: 'bore,stroke,cylinders'")

    bore = parse_fraction(parts[0])
    stroke = parse_fraction(parts[1])
    cylinders = int(parts[2])

    bore_mm = bore * 25.4
    stroke_mm = stroke * 25.4
    print(bore_mm, stroke_mm)

    displacement = (math.pi / 4) * bore**2 * stroke * cylinders
    return displacement

# Example usage
input_str = "4 7/8,6,4"
disp = calculate_displacement(input_str)
print(f"{disp:.1f}")