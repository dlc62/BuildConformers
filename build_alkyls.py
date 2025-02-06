#!/usr/bin/python3

import sys
import operator
import numpy as np
from copy import deepcopy as cp

def tadd(t1,t2):
  result = tuple(map(operator.add,t1,t2))
  return result

def grow(OldC,Hydrogens,Verts,index):

  # Choose newC from the last set of hydrogens
  NewC = Hydrogens.pop(index) 
  # Sprout new hydrogens
  NewV = [tadd(V,NewC) for V in Verts] 
  # Remove the hydrogen that lies along C-C bond being formed
  NewV.remove(OldC)

  return NewC, NewV

def dist_mat2(coords):
 
  x2 = np.sum(coords**2,axis=1)
  y2 = x2[:,np.newaxis]
  xy = np.dot(coords,coords.T)
  r2 = x2 + y2 - 2*xy
  zeros = np.zeros_like(r2)

  return np.maximum(r2,zeros)

def make_conformers(ChainLength):
#--------------------------------------------------------
# Construct alkyl chains
#--------------------------------------------------------
# Data: Vertex coordinates
  Vodd = [(-1,-1,-1),(-1,1,1),(1,-1,1),(1,1,-1)]
  Veven = [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]

# Start off all chains the same way
# Place C1 at the origin, and hydrogens around it
  Carbons = [(0,0,0)]
  Hydrogens = [Veven[:]]

# Arbitrarily grow alkyl chain in the direction of 
# the first H atom attached to the first C atom
  NewC, NewV = grow(Carbons[-1],Hydrogens[-1],Vodd,0)
  Carbons.append(NewC); Hydrogens.append(NewV)

# And again but this time attaching C3 to C2 
  NewC, NewV = grow(Carbons[-1],Hydrogens[-1],Veven,0)
  Carbons.append(NewC); Hydrogens.append(NewV)

# Generate anti and gauche conformations of C4
# Spawning new chains in the process
  OldChains = [[Carbons,Hydrogens]]; NewChains = []
  for [Carbons,Hydrogens] in OldChains:
    for index in range(0,2):
      NewCarbons = cp(Carbons); NewHydrogens = cp(Hydrogens)
      NewC, NewV = grow(NewCarbons[-1],NewHydrogens[-1],Vodd,index)
      NewChains.append([NewCarbons+[NewC],NewHydrogens+[NewV]])
  OldChains = cp(NewChains)

# Grow in all possible directions until intended chain length reached
# Discard chains that result in steric clashes (overlapping H atoms)
  parity = 1
  for i in range(4,ChainLength):
    if parity == 1: 
      Verts = Veven
    else: 
      Verts = Vodd
    NewChains = []
    for [Carbons,Hydrogens] in OldChains:
      AllHydrogens = [Atom for Group in Hydrogens for Atom in Group]
      for index in range(0,3):
        NewCarbons = cp(Carbons); NewHydrogens = cp(Hydrogens)
        NewC, NewV = grow(NewCarbons[-1],NewHydrogens[-1],Verts,index)
        if len(set(NewV) & set(AllHydrogens)) == 0:
          NewChains.append([NewCarbons+[NewC],NewHydrogens+[NewV]])
    parity *= -1
    OldChains = cp(NewChains)
  print('Number of alkyl chains =', len(NewChains))

#------------------------------------------------------------------------
# Customise alkyl chains in united atom representation 
# - add carboxylate group, rotate, scale coordinates, compute energies
#------------------------------------------------------------------------
# Construct rotation matrix to align first C-C bond with z axis
# Uses Euler-Rodrigues rotation formula
  z = np.array([0,0,1]); I = np.eye(3,3) 
  b = np.array(Veven[0])/np.sqrt(3)
  v = np.cross(b,z)
  c = np.dot(b,z) 
  vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
  R = I + vx + np.dot(vx,vx)/(1+c)

# Work out where terminal oxygens are going to go on rotated chains
# Note that there will be three different orientations
  Scale = 0.82  # C-O bond is 82 % shorter than C-C bond
  TermH = [Scale * np.dot(R,np.array(Vert)) for Vert in Veven[1:]]
  TermO = [[[x,y,z],[-x,-y,z]] for [x,y,z] in TermH]

# Set up Lennard-Jones parameters controlling non-bonded interactions
# There are no intramolecular electrostatic interactions and torsional
# destabilisation is purely steric in origin (no explicit torsional terms).
# UNITS: (kJ/mol nm^X)^{1/2} where X = 6 or 12, respectively
# Hard-coded for primary organic acids only
  NB_C6  = {'O': 0.04756,   'C1': 0.04838,  'C2': 0.08642,  'C3': 0.09805}
  NB_C12 = {'O': 0.8611E-3, 'C1': 2.222E-3, 'C2': 5.828E-3, 'C3': 5.162E-3} 
  AtomType = ['O','O','C1'] + ['C2']*(ChainLength-2) + ['C3']
  Half_C6  = np.array([NB_C6[Atom] for Atom in AtomType]) 
  Half_C12 = np.array([NB_C12[Atom] for Atom in AtomType]) 
# Construct unscaled C6 and C12 coefficient matrices
  C6 = np.outer(Half_C6,Half_C6); C12 = np.outer(Half_C12,Half_C12)
# Scale down torsional interaction parameters for alkyl carbon interactions
  Tors_C6  = {'C2':0.06783,   'C3':0.08278}
  Tors_C12 = {'C2':2.178E-3, 'C3':2.456E-3}
  for i in range(3,ChainLength+2-4):
    j = i+3
    C6[i,j]  = Tors_C6[AtomType[i]]*Tors_C6[AtomType[j]]
    C12[i,j] = Tors_C12[AtomType[i]]*Tors_C12[AtomType[j]]
# Zero out interaction parameters for 1,2- and 1,3-bonded atoms 
# And prevent double-counting
  for i in range(0,2):
    for j in range(0,4):
      C6[i,j] = 0.0
      C12[i,j] = 0.0
  for i in range(2,ChainLength+2-3):
    for j in range(0,i+3):
      C6[i,j] = 0.0
      C12[i,j] = 0.0
  for i in range(ChainLength+2-3,ChainLength+2):
    for j in range(0,ChainLength+2):
      C6[i,j] = 0.0
      C12[i,j] = 0.0

# Build rotated molecules, scale and compute energies
  Scale_to_nm = 1.54/np.sqrt(3)/10.0
  Energies = []; Molecules = []
  for [Carbons,Hydrogens] in NewChains:
    RotC = [np.dot(R,np.array(C)) for C in Carbons]
    for Oxygens in TermO:
      Molecule = np.array(Oxygens + RotC) * Scale_to_nm
      R2 = dist_mat2(Molecule)
      R2_inv = np.zeros_like(R2)
      for i in range(0,len(R2)):
        for j in range(i+1,len(R2)):
          R2_inv[i,j] = 1/R2[i,j]
      R6_inv = R2_inv**3
      R12_inv = R6_inv**2
      E_mat = -C6*R6_inv + C12*R12_inv
      E = np.sum(E_mat)
    # Scale molecule to Angstroms and shift so z = 0 for O atoms
      Molecule *= 10
      Zoff = - Molecule[0,2]
      Molecule[:,2] += Zoff
      Molecules.append(Molecule)
      Energies.append(E)

  print('C6 =', C6)
  print('C12 =', C12)
  print('E_mat =', E_mat)
  print('Number of conformers = ', len(Molecules))
  print('Energies =', Energies)

  return AtomType,Molecules,Energies

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print('Usage: build_alkyls.py <ChainLength>')
    sys.exit()
  else:
    ChainLength = int(sys.argv[1])
    Atoms,Molecules,Energies = make_conformers(ChainLength)
