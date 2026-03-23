

# I. `INCAR` generation

# II. SCF calc for general metals

```
cat > INCAR <<!
# ===== Initialization =====
ISTART = 0
ICHARG = 2

# ===== K-points / XC =====
KSPACING = 0.2
KGAMMA = .TRUE.
GGA = PE

# ===== Electronic =====
ENCUT = 600
ALGO = Normal
NELM = 120
EDIFF = 1E-06
ISMEAR = 0
SIGMA = 0.02
PREC = Accurate

# ===== Ionic =====
NSW = 1
IBRION = -1

# ===== Output =====
LCHARG = .FALSE.
LWAVE  = .FALSE.

# ===== Performance =====
LREAL = A
!
```
