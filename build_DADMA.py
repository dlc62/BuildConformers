#!/usr/bin/python3

import sys
import operator
import numpy as np
from copy import deepcopy as cp

def tadd(t1,t2):
  result = tuple(map(operator.add,t1,t2))
  return result

def tminus(t1,t2):
  result = tuple(map(operator.sub,t1,t2))
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
  AllylEven = [ [ [(1,1,1),(1/3,7/3,5/3)] , [[(7/3,1/3,5/3)],[(1,3,3),(-1,3,1)]] ],
                [ [(1,1,1),(7/3,1/3,5/3)] , [[(1/3,7/3,5/3)],[(3,1,3),(3,-1,1)]] ],
            [ [(1,-1,-1),(5/3,-7/3,-1/3)] , [[(5/3,-1/3,-7/3)],[(3,-3,-1),(1,-3,1)]] ],
            [ [(1,-1,-1),(5/3,-1/3,-7/3)] , [[(5/3,-7/3,-1/3)],[(3,-1,-3),(1,1,-3)]] ],
            [ [(-1,1,-1),(-1/3,7/3,-5/3)] , [[(-7/3,1/3,-5/3)],[(-1,3,-3),(1,3,-1)]] ],
            [ [(-1,1,-1),(-7/3,1/3,-5/3)] , [[(7/3,-1/3,-5/3)],[(-3,1,-3),(-3,-1,-1)]] ],
            [ [(-1,-1,1),(-5/3,-7/3,1/3)] , [[(-5/3,-1/3,7/3)],[(-3,-3,1),(-1,-3,-1)]] ],
            [ [(-1,-1,1),(-5/3,1/3,-7/3)] , [[(-5/3,-7/3,1/3)],[(-3,-1,3),(-1,1,3)]] ] ]
  AllylOdd  = [ [ [(-1,-1,-1),(-1/3,-7/3,-5/3)],[[(-7/3,-1/3,-5/3)],[(-1,-3,-3),(1,-3,-1)]] ],
                [ [(-1,-1,-1),(-7/3,-1/3,-5/3)],[[(-1/3,-7/3,-5/3)],[(-3,-1,-3),(-3,1,-1)]] ],
                [ [(-1,1,1),(-5/3,7/3,1/3)],[[(-5/3,1/3,7/3)],[(-3,3,1),(-1,3,-1)]] ],
                [ [(-1,1,1),(-5/3,1/3,7/3)],[[(-5/3,7/3,1/3)],[(-3,1,3),(-1,-1,3)]] ],
                [ [(1,-1,1),(1/3,-7/3,5/3)],[[(7/3,-1/3,5/3)],[(1,-3,3),(-1,-3,1)]] ],
                [ [(1,-1,1),(7/3,-1/3,5/3)],[[(1/3,-7/3,5/3)],[(3,-1,3),(3,1,1)]] ],
                [ [(1,1,-1),(5/3,7/3,-1/3)],[[(5/3,1/3,-7/3)],[(3,3,-1),(1,3,1)]] ],
                [ [(1,1,-1),(5/3,1/3,-7/3)],[[(5/3,7/3,-1/3)],[(3,1,-3),(1,-1,-3)]] ] ]

# Start off all chains the same way
# Place C1 (=> N) at the origin, and hydrogens around it
  Carbons = [(0,0,0)]
  Hydrogens = [Veven[:]]

# Place first two methyl groups
  for i in range(0,2):
    NewC, NewV = grow(Carbons[0],Hydrogens[0],Vodd,0)
    Carbons.append(NewC); Hydrogens.append(NewV)

# Build chains one by one keeping all sterically-allowed conformers
  OldChains = cp([[Carbons,Hydrogens]])
  for i in range(0,2):

    # First methyl is special - no conformational flexibility
    for [Carbons,Hydrogens] in OldChains:
      NewC, NewV = grow(Carbons[0],Hydrogens[0],Vodd,0)
      Carbons.append(NewC); Hydrogens.append(NewV)

    # Now start creating different conformers
    parity = 1
    for i in range(1,ChainLength-2):
      if parity == 1: Verts = Veven
      else: Verts = Vodd
      NewChains = []
      for [Carbons,Hydrogens] in OldChains:
        AllHydrogens = [Atom for Group in Hydrogens for Atom in Group]
        if len(set(Carbons) & set(AllHydrogens)) == 0:
          for index in range(0,3):
            NewCarbons = cp(Carbons); NewHydrogens = cp(Hydrogens)
            NewC, NewV = grow(NewCarbons[-1],NewHydrogens[-1],Verts,index)
            if len(set(NewV) & set(AllHydrogens)) == 0:
              NewChains.append([NewCarbons+[NewC],NewHydrogens+[NewV]])
      parity *= -1
      OldChains = cp(NewChains)

    # New bit of code to put terminal allyls on each chain
    parity *= -1
    if parity == 1: Fragments = AllylEven
    else: Fragments = AllylOdd
    NewChains = []
    for [Carbons,Hydrogens] in OldChains:
      OldC = Carbons[-1]; OldH = Hydrogens[-1]
      for H in OldH:
        vec = tminus(H,OldC)
        for Fragment in Fragments:
          NewCarbons = cp(Carbons); NewHydrogens = cp(Hydrogens)
          if vec == Fragment[0][0]:
            NewHydrogens[-1].remove(H)
            NewC = [tadd(FragC,OldC) for FragC in Fragment[0]]
            NewH = [[tadd(Atom,OldC) for Atom in FragH] for FragH in Fragment[1]] 
            AllH = [Atom for Group in NewHydrogens+NewH for Atom in Group]
            if len(set(NewCarbons+NewC) & set(AllH)) == 0:
              NewChains.append([NewCarbons+NewC,NewHydrogens+NewH])
    OldChains = cp(NewChains)
 
#------------------------------------------------------------------------
# Customise alkylammonium chains in united atom representation 
# - add nitrogen atom, rotate, scale coordinates, compute energies
#------------------------------------------------------------------------
# Construct rotation matrix to align head group with z axis
# Uses Euler-Rodrigues rotation formula
  z = np.array([0,0,1]); I = np.eye(3,3) 
  b = np.array(tadd(Veven[0],Veven[1]))
  b = b/np.linalg.norm(b)
  v = np.cross(b,z)
  c = np.dot(b,z) 
  vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
  R = I + vx + np.dot(vx,vx)/(1+c)

# Set up Lennard-Jones parameters controlling non-bonded interactions
# There are no intramolecular electrostatic interactions and torsional
# destabilisation is purely steric in origin (no explicit torsional terms).
# UNITS: (kJ/mol nm^X)^{1/2} where X = 6 or 12, respectively
# Hard-coded for quaternary alkylamines with two allyl-terminated longer chains 
# Minimum chain length = 3
  Head = ['N','CN','CN']
  Chain = ['CN'] + ['C2']*(ChainLength-3) + ['C1','C2']
  ChainH = ['CN'] + ['C2']*(ChainLength-3) + ['C1','HC1','C2','HC2','HC2']
  AtomType = Head + Chain + Chain
  ExtendedAtomType = Head + ChainH + ChainH
#  print(len(AtomType))

  ChainIndices = []
  for i in range(0,2):
    ChainInd = [3 + i*ChainLength + j for j in range(0,ChainLength)]
    ChainIndices += [ChainInd]

  Bonded = [[0,1],[0,2]]
  Tors = []
  for ChainInd in ChainIndices:
    Tors += [[0,ChainInd[2]],[1,ChainInd[1]],[2,ChainInd[1]]]
    Bonded += [[0,ChainInd[0]],[0,ChainInd[1]],[1,ChainInd[0]],[2,ChainInd[0]]]
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
  NB_C6    = {'N': 0.04936,  'CN': 0.04838,  'C1': 0.07790, 'C2': 0.08642, 'C3': 0.09805}
  NB_C12   = {'N': 1.523E-3, 'CN': 2.222E-3, 'C1': 9.850E-3,'C2': 5.828E-3,'C3': 5.162E-3} 
  Tors_C6  = {'N': 0.04936,  'CN': 0.04838,  'C1': 0.05396, 'C2':0.06783,  'C3':0.08278}
  Tors_C12 = {'N': 1.523E-3, 'CN': 2.222E-3, 'C1': 1.933E-3,'C2':2.178E-3, 'C3':2.456E-3}
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

# Build rotated molecules, shift so that terminal Cs are at z = 0
# Scale to nm and compute energies, then back to Angstrom for output
  Scale_to_nm = 1.54/np.sqrt(3)/10.0
  Scale_to_Ang = 1.54/np.sqrt(3)
  Scale_CH = 0.667
  Energies = []; Molecules = []
  for [Carbons,Hydrogens] in NewChains:
    # Rotate
    RotC = [np.dot(R,np.array(C)) for C in Carbons]
    # Shift
    Off = np.array([0,0,RotC[1][-1]]) 
    TransRotC = [C-Off for C in RotC] 
    TransRotH = [[np.dot(R,np.array(H))-Off for H in Group] for Group in Hydrogens] 
    Molecule = np.array(RotC) 
    # Scale (C atoms only)
    Molecule = np.array(TransRotC)
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
    Energies.append(E)
    # Construct molecule with all C atoms and allyl hydrogens
    # Scale to Angstrom and store (note that C-H bonds will be a little too long)
    Molecule = [C for C in TransRotC[0:3]]
    for ChainInd in ChainIndices:
      for Index in ChainInd[0:-2]:
        Molecule += [TransRotC[Index]] 
      for Index in ChainInd[-2:]:
        C = TransRotC[Index]
        H = [((H-C)*Scale_CH + C) for H in TransRotH[Index]]
        Molecule += [C]
        Molecule += H
    Molecules.append(np.array(Molecule)*Scale_to_Ang)

#  print('Last molecule = ', Molecule)
#  print('Natom = ', len(Molecule))
#  print('Atom types = ', ExtendedAtomType)
#  print('Natom = ', len(ExtendedAtomType))
#  output = [at + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n' for at,[x,y,z] in zip(ExtendedAtomType,Molecule)]
#  with open('mol.xyz','w') as f:
#    f.writelines(output)
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

  return ExtendedAtomType,Molecules,Energies

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print('Usage: build_aliquat.py <ChainLength>')
    sys.exit()
  else:
    ChainLength = int(sys.argv[1])
    Atoms,Molecules,Energies = make_conformers(ChainLength)
