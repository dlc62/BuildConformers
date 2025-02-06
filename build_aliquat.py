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
# Place C1 (=> N) at the origin, and hydrogens around it
  Carbons = [(0,0,0)]
  Hydrogens = [Veven[:]]

# Place first solo methyl group
  NewC, NewV = grow(Carbons[0],Hydrogens[0],Vodd,0)
  Carbons.append(NewC); Hydrogens.append(NewV)

# Build chains one by one keeping all sterically-allowed conformers
  OldChains = cp([[Carbons,Hydrogens]])
  for i in range(0,3):
    # First methyl is special - no conformational flexibility
    for [Carbons,Hydrogens] in OldChains:
      NewC, NewV = grow(Carbons[0],Hydrogens[0],Vodd,0)
      Carbons.append(NewC); Hydrogens.append(NewV)
    # Now start creating different conformers
    parity = 1
    for i in range(1,ChainLength):
      if parity == 1: Verts = Veven
      else: Verts = Vodd
      if i == ChainLength-1: imax = 1 # Cheat and only add a single C on the last pass
      else: imax = 3
      imax = 3
      NewChains = []
      for [Carbons,Hydrogens] in OldChains:
        AllHydrogens = [Atom for Group in Hydrogens for Atom in Group]
        if len(set(Carbons) & set(AllHydrogens)) == 0:
          for index in range(0,imax):
            NewCarbons = cp(Carbons); NewHydrogens = cp(Hydrogens)
            NewC, NewV = grow(NewCarbons[-1],NewHydrogens[-1],Verts,index)
            if (len(set(NewV) & set(AllHydrogens)) == 0) or (imax == 1):
              NewChains.append([NewCarbons+[NewC],NewHydrogens+[NewV]])
      parity *= -1
      OldChains = cp(NewChains)

#  print('All hydrogens', AllHydrogens)
#  print('Last Chain Carbons', NewChains[-1][0])
#  print('Carbons in common with AllHydrogens', set(Carbons) & set(AllHydrogens))
#  print('New vertices in common with AllHydrogens', set(NewV) & set(AllHydrogens))
#  print('Number of alkyl chains =', len(NewChains))
    
#------------------------------------------------------------------------
# Customise alkylammonium chains in united atom representation 
# - add nitrogen atom, rotate, scale coordinates, compute energies
#------------------------------------------------------------------------
# Construct rotation matrix to align first (N-C) bond with z axis
# Uses Euler-Rodrigues rotation formula
  z = np.array([0,0,1]); I = np.eye(3,3) 
  b = np.array(Veven[0])/np.sqrt(3)
  v = np.cross(b,z)
  c = np.dot(b,z) 
  vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
  R = I + vx + np.dot(vx,vx)/(1+c)

# Set up Lennard-Jones parameters controlling non-bonded interactions
# There are no intramolecular electrostatic interactions and torsional
# destabilisation is purely steric in origin (no explicit torsional terms).
# UNITS: (kJ/mol nm^X)^{1/2} where X = 6 or 12, respectively
# Hard-coded for quaternary alkylamines with three longer chains only 
  Head = ['N','CN']
  Chain = ['CN'] + ['C2']*(ChainLength-2) + ['C3']
  AtomType = Head + Chain + Chain + Chain
#  print(len(AtomType))

  ChainIndices = []
  for i in range(0,3):
    ChainInd = [2 + i*ChainLength + j for j in range(0,ChainLength)]
    ChainIndices += [ChainInd]

  Bonded = [[0,1]]
  Tors = []
  for ChainInd in ChainIndices:
    Tors += [[0,ChainInd[2]],[1,ChainInd[1]]]
    Bonded += [[0,ChainInd[0]],[0,ChainInd[1]],[1,ChainInd[0]]]
    for i in range(0,ChainLength-3):
      Tors += [[ChainInd[i],ChainInd[i+3]]]
      for j in range(i+1,min(i+3,ChainLength)):
        Bonded += [[ChainInd[i],ChainInd[j]]]
    for i in range(ChainLength-3,ChainLength-1):
      for j in range(i+1,ChainLength):
        Bonded += [[ChainInd[i],ChainInd[j]]]
#  print(Bonded)
#  print(Tors)
# Force field parameters
  NB_C6    = {'N': 0.04936,  'CN': 0.04838,  'C2': 0.08642,  'C3': 0.09805}
  NB_C12   = {'N': 1.523E-3, 'CN': 2.222E-3, 'C2': 5.828E-3, 'C3': 5.162E-3} 
  Tors_C6  = {'N': 0.04936,  'CN': 0.04838,  'C2':0.06783,   'C3':0.08278}
  Tors_C12 = {'N': 1.523E-3, 'CN': 2.222E-3, 'C2':2.178E-3, 'C3':2.456E-3}
  Half_C6  = np.array([NB_C6[Atom] for Atom in AtomType]) 
  Half_C12 = np.array([NB_C12[Atom] for Atom in AtomType]) 
# Construct unscaled C6 and C12 coefficient matrices
  C6 = np.triu(np.outer(Half_C6,Half_C6),1); C12 = np.triu(np.outer(Half_C12,Half_C12),1)
# Zero out interactions for bonded atoms
  for [i,j] in Bonded:
    C6[i,j] = 0.0
    C12[i,j] = 0.0
# Scale down torsional interactions along alkyl chain
  for i,j in Tors:
    C6[i,j]  = Tors_C6[AtomType[i]]*Tors_C6[AtomType[j]]
    C12[i,j] = Tors_C12[AtomType[i]]*Tors_C12[AtomType[j]]
#  print('C6 = ', C6)
#  print('C12 = ', C12)

# Build rotated molecules, shift so that terminal C is at origin
# Scale to nm and compute energies, then back to Angstrom for output
  Scale_to_nm = 1.54/np.sqrt(3)/10.0
  Energies = []; Molecules = []
  for [Carbons,Hydrogens] in NewChains:
    # Rotate
    RotC = [np.dot(R,np.array(C)) for C in Carbons]
    Molecule = np.array(RotC) 
    # Shift
    Molecule[:,:] -= Molecule[1,:]
    # Scale
    Molecule *= Scale_to_nm
    # Compute non-bonded distances (R^-6 and R^-12)
    R2 = dist_mat2(Molecule)
    R2_inv = np.zeros_like(R2)
    for i in range(0,len(R2)):
      for j in range(i+1,len(R2)):
        R2_inv[i,j] = 1/R2[i,j]
    R6_inv = np.triu(R2_inv**3,1)
    R12_inv = np.triu(R6_inv**2,1)
    # Contract with C_6 and C_12 coefficient matrices to get pairwise
    # contributions to torsional through-space and non-bonded interaction energies
    E_mat = -C6*R6_inv + C12*R12_inv
    E = np.sum(E_mat)
    # Scale back to Angstrom and store
    Molecule *= 10
    Molecules.append(Molecule)
    Energies.append(E)

#  print('C6 =', C6)
#  print('C12 =', C12)
#  print('R2 =', R2)
#  print('R6_inv', R6_inv)
#  print('R12_inv', R12_inv)
#  print('C12*R12_inv', C12*R12_inv)
#  print('E_mat =', E_mat)
#  print(AtomType)
#  print(Molecule)
#  print('R2 = ', R2)
  print('Number of conformers = ', len(Molecules))
  print('Energies =', Energies)

  return AtomType,Molecules,Energies

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print('Usage: build_aliquat.py <ChainLength>')
    sys.exit()
  else:
    ChainLength = int(sys.argv[1])
    Atoms,Molecules,Energies = make_conformers(ChainLength)
