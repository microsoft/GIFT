"""
Extract 267-dimensional features for external compounds
EXACTLY matching extract_graph_for_gnn.py methodology
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

def get_compound_features_267(smiles: str) -> np.ndarray:
    """
    Extract 267-dimensional features matching extract_graph_for_gnn.py
    
    Returns:
        267-dim array: [11 basic descriptors, 256 Morgan fingerprint]
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 1. Basic molecular descriptors (11 features) - EXACT match to extract_graph_for_gnn.py
        basic_features = [
            Descriptors.MolWt(mol),                    # molecular weight
            Descriptors.MolLogP(mol),                  # logP
            Descriptors.NumHAcceptors(mol),            # H-bond acceptors
            Descriptors.NumHDonors(mol),               # H-bond donors
            Descriptors.NumRotatableBonds(mol),        # rotatable bonds
            Descriptors.NumAromaticRings(mol),         # aromatic rings
            Descriptors.TPSA(mol),                     # polar surface area
            Descriptors.HeavyAtomCount(mol),           # heavy atoms
            Descriptors.qed(mol),                      # QED score
            Descriptors.RingCount(mol),                # total rings
            Descriptors.NumAliphaticRings(mol)         # aliphatic rings
        ]
        
        # 2. Morgan fingerprint (256 features) - EXACT parameters as extract_graph_for_gnn.py
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
        fp = mfpgen.GetFingerprint(mol)
        
        # Convert to numpy array
        fp_array = np.zeros(256, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        # Combine: 11 + 256 = 267 dimensions
        combined_features = np.concatenate([
            np.array(basic_features, dtype=np.float32),
            fp_array
        ])
        
        return combined_features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def test_feature_extraction():
    """Test that we get exactly 267 dimensions"""
    # Test with aspirin
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    features = get_compound_features_267(test_smiles)
    
    if features is not None:
        print(f"Feature extraction successful")
        print(f"Shape: {features.shape}")
        print(f"Expected: (267,)")
        print(f"Match: {features.shape == (267,)}")
        print(f"\nFirst 11 features (basic descriptors):")
        print(features[:11])
        print(f"\nFingerprint starts at index 11:")
        print(f"fp[0:5] = {features[11:16]}")
        print(f"Fingerprint sparsity: {np.sum(features[11:] == 0)/256*100:.1f}% zeros")
    else:
        print("Feature extraction failed")


if __name__ == "__main__":
    test_feature_extraction()
