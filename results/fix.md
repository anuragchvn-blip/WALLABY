# ❌ **STILL FAILED - Actually Got WORSE**

## Comparison: Run 1 vs Run 2

```
                    RUN 1        RUN 2       Expected
─────────────────────────────────────────────────────────
v_pec range:    ±2000 km/s   -3000/+2942   ±500-1500
v_pec mean:      1909 km/s    2852 km/s    ~100-300
v_pec median:    (unknown)    -3000 km/s   ~0-200
Galaxies kept:   195 (2%)     168 (1.7%)   ~7000 (70%)
Rejected:        9804 (98%)   9831 (98.3%) ~3000 (30%)
```

### 🚨 **SMOKING GUN: Median = -3000.0**

**This is EXACTLY your lower cutoff value!**

This means:
- **50%+ of galaxies have v_pec < -3000 km/s**
- You're just truncating a completely wrong distribution
- Widening the cutoff didn't fix anything - it just kept more garbage

---

## **What This Distribution Means:**

**Healthy data should look like:**
```
      │
  800 │     ╱‾╲
  600 │    ╱   ╲
  400 │   ╱     ╲___
  200 │  ╱          ╲___
    0 │_╱_______________╲___
      -500  0  +500 +1000  km/s
      
Mean ≈ +200, Median ≈ +150, Std ≈ 400
```

**Your data looks like:**
```
      │
 5000 │|
 4000 │|
 3000 │|              ╱╲
 2000 │|             ╱  ╲
 1000 │|____________╱____╲
    0 │
     -3000    0   +2500  km/s
     (cutoff!)    
     
Median = -3000 (pile-up at cutoff!)
Mean = +2852 (tail pulling it positive)
```

This is **catastrophically wrong**.

---

## **The Bug is 100% in Peculiar Velocity Calculation**

### **Likely Issues:**

**1. Sign Error:**
```python
# WRONG:
v_pec = H0 * D - c * z_obs  # Backwards!

# CORRECT:
v_pec = c * z_obs - H0 * D
```

**2. Units Mismatch:**
```python
# If D is in kpc instead of Mpc:
H0 = 70  # km/s/Mpc
D = 16500  # kpc (Virgo), not 16.5 Mpc!
v_pec = c*z - H0*16500  # HUGE negative number!

# Should be:
D = 16.5  # Mpc
v_pec = c*z - H0*16.5  # Reasonable
```

**3. Redshift in Wrong Units:**
```python
# If z is stored as velocity already:
z_obs = 3500  # Already km/s, not dimensionless!
v_pec = c * 3500 - H0*D  # c*3500 = 1 billion km/s!

# Should be:
z_obs = 0.0117  # Dimensionless
v_pec = c * 0.0117 - H0*D  # = 3500 - ...
```

---

## **STOP - Debug Before Running Again**

### **Required Diagnostic Output:**

Add this to your code **before quality cuts:**

```python
print("\n=== PECULIAR VELOCITY DIAGNOSTICS ===")
print(f"First 20 galaxies:")
print(f"{'i':>3} {'RA':>8} {'Dec':>8} {'z':>10} {'D(Mpc)':>10} "
      f"{'v_H':>10} {'v_obs':>10} {'v_pec':>10}")
print("-" * 80)

for i in range(min(20, len(galaxies))):
    ra = galaxies[i]['ra']
    dec = galaxies[i]['dec']
    z = galaxies[i]['z']
    D = galaxies[i]['distance']
    v_H = H0 * D
    v_obs = c * z
    v_pec = galaxies[i]['v_pec']
    
    print(f"{i:3d} {ra:8.3f} {dec:8.3f} {z:10.6f} {D:10.3f} "
          f"{v_H:10.1f} {v_obs:10.1f} {v_pec:10.1f}")

print(f"\nFull distribution:")
print(f"  v_pec min:     {np.min(v_pec_all):10.1f} km/s")
print(f"  v_pec 5%ile:   {np.percentile(v_pec_all, 5):10.1f}")
print(f"  v_pec median:  {np.median(v_pec_all):10.1f}")
print(f"  v_pec mean:    {np.mean(v_pec_all):10.1f}")
print(f"  v_pec 95%ile:  {np.percentile(v_pec_all, 95):10.1f}")
print(f"  v_pec max:     {np.max(v_pec_all):10.1f}")
print(f"  v_pec std:     {np.std(v_pec_all):10.1f}")

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(v_pec_all, bins=100, range=(-5000, 5000), alpha=0.7)
plt.axvline(0, color='r', linestyle='--', label='v=0')
plt.xlabel('v_pec (km/s)')
plt.ylabel('Count')
plt.title('Peculiar Velocity Distribution (BEFORE quality cuts)')
plt.legend()
plt.savefig('vpec_distribution_raw.png')
print("Saved: vpec_distribution_raw.png")
```

---

## **What You Should See (If Fixed):**

```
=== PECULIAR VELOCITY DIAGNOSTICS ===
First 20 galaxies:
  i       RA      Dec          z    D(Mpc)        v_H      v_obs      v_pec
--------------------------------------------------------------------------------
  0  180.500   12.300   0.003500      14.5      1015.0     1049.6       34.6
  1  195.200  -23.400   0.008200      35.2      2464.0     2458.4       -5.6
  2  210.100   45.600   0.015300      67.8      4746.0     4586.7     -159.3
  ...

Full distribution:
  v_pec min:        -745.2 km/s
  v_pec 5%ile:      -320.5
  v_pec median:       85.3
  v_pec mean:        215.7
  v_pec 95%ile:      890.2
  v_pec max:        1523.8
  v_pec std:         425.3
```

**Histogram should be roughly Gaussian, centered near 0-300 km/s**

---

## **Current Status: CANNOT PROCEED**

You have a **pipeline that works** but is being fed **completely wrong input data**.

It's like having a perfect Ferrari with sugar in the gas tank.

### **DO NOT:**
- ❌ Run more reconstructions
- ❌ Adjust parameters
- ❌ Try different algorithms
- ❌ Change quality thresholds

### **DO:**
1. ✅ Add diagnostic output above
2. ✅ **Share the first 20 rows** with me
3. ✅ Show the v_pec histogram
4. ✅ Let me identify the exact bug
5. ✅ Fix the calculation
6. ✅ **THEN** re-run

---

## **My Bet:**

Based on median = -3000 and mean = +2852, I predict:

**Either:**
- Distance is in kpc, not Mpc (1000× error)
- Redshift is already velocity in km/s
- Sign is flipped
- Wrong column being read from catalog
