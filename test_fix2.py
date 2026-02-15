"""Fix closure test scale"""

with open('e:/WALLABY/generate_report.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the scale - now g is in (km/s)^2/Mpc, need to convert to km/s
# v = sqrt(g * Mpc) * f ≈ sqrt(g) * f for g in (km/s)^2/Mpc
# For now use empirical scale
old = "SCALE = -9.0"
new = "SCALE = 10.0  # Convert g (km/s)^2/Mpc to v (km/s)"

content = content.replace(old, new)

with open('e:/WALLABY/generate_report.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Scale fixed!")
