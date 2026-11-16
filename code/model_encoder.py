import warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import json
import pickle
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve, precision_recall_curve, f1_score
from matplotlib.patches import Patch

import argparse
from datetime import datetime
from collections import defaultdict, Counter

# SEED FOR REPRODUCIBILITY
SEED = 42

def validate_soft_edges_on_test_data(model, x, edge_index, edge_attr, node_types, test_pairs, data, results_dir="results"):
    """
    Validate if the soft edges drawn by the model on test data correspond to actual positive labels.
    This tests the model's ability to identify true interactions without being told!
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    print("\n" + "="*80)
    print("SOFT EDGE VALIDATION ON TEST DATA")
    print("="*80)
    print("Testing if model-selected soft edges correspond to real drug-target interactions")
    
    model.eval()
    device = x.device if hasattr(x, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert test pairs to a set for fast lookup
    test_positive_set = set(test_pairs)
    
    with torch.no_grad():
        # Get node types on GPU
        if isinstance(node_types, np.ndarray):
            node_types_gpu = torch.from_numpy(node_types).to(device)
        else:
            node_types_gpu = node_types
        
        # Process through GNN to get embeddings
        compound_mask = node_types_gpu == 0
        protein_mask = node_types_gpu == 1
        
        # Get intermediate embeddings (before final layer) for edge masking
        x_current = x
        for i, (conv, bn) in enumerate(zip(model.convs[:-1], model.bns)):
            x_current = conv(x_current, edge_index, edge_attr)
            x_current = bn(x_current)
            x_current = F.relu(x_current)
        
        intermediate_x = x_current
        
        # Extract embeddings
        compound_embeddings = intermediate_x[compound_mask]
        protein_embeddings = intermediate_x[protein_mask]
        num_compounds = data['num_compounds']
        num_proteins = data['num_proteins']
        
        # CRITICAL: Get attention scores for ALL possible C-P pairs
        print(f"\nComputing attention scores for all {num_compounds * num_proteins} possible C-P pairs...")
        full_attention_scores = model.edge_mask.compute_attention_scores(
            compound_embeddings, protein_embeddings
        )
        
        # Flatten scores and get indices
        flat_scores = full_attention_scores.flatten()
        
        # Create labels: 1 if pair is in test set, 0 otherwise
        labels = torch.zeros_like(flat_scores)
        scores_list = []
        labels_list = []
        pairs_list = []
        
        for c_idx in range(num_compounds):
            for p_idx in range(num_proteins):
                flat_idx = c_idx * num_proteins + p_idx
                score = flat_scores[flat_idx].item()
                is_positive = (c_idx, p_idx) in test_positive_set
                
                scores_list.append(score)
                labels_list.append(1 if is_positive else 0)
                pairs_list.append((c_idx, p_idx))
        
        scores_array = np.array(scores_list)
        labels_array = np.array(labels_list)
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(scores_array)[::-1]
        sorted_scores = scores_array[sorted_indices]
        sorted_labels = labels_array[sorted_indices]
        sorted_pairs = [pairs_list[i] for i in sorted_indices]
        
        print(f"\nValidation Results:")
        print(f"Total possible C-P pairs: {len(scores_list)}")
        print(f"Actual positive pairs in test set: {np.sum(labels_array)} ({np.sum(labels_array)/len(labels_array)*100:.2f}%)")
        
        # Calculate precision at different top-k values
        k_values = [10, 25, 50, 100, 200, 500, 1000]
        precisions = []
        recalls = []
        
        for k in k_values:
            if k <= len(sorted_labels):
                top_k_labels = sorted_labels[:k]
                true_positives = np.sum(top_k_labels)
                precision = true_positives / k
                recall = true_positives / np.sum(labels_array) if np.sum(labels_array) > 0 else 0
                precisions.append(precision)
                recalls.append(recall)
                print(f"  Top-{k:4d}: Precision={precision:.3f}, Recall={recall:.3f}, TP={true_positives}/{k}")
        
        # Calculate average precision score
        avg_precision = average_precision_score(labels_array, scores_array)
        print(f"\nAverage Precision Score: {avg_precision:.4f}")
        
        # Random baseline
        random_precision = np.sum(labels_array) / len(labels_array)
        print(f"Random baseline precision: {random_precision:.4f}")
        print(f"Improvement over random: {avg_precision/random_precision:.2f}x")
        
        # Show top scoring pairs
        print(f"\n{'='*60}")
        print("TOP 20 SOFT EDGE PREDICTIONS:")
        print(f"{'='*60}")
        print(f"{'Rank':<6} {'C_idx':<8} {'P_idx':<8} {'Score':<10} {'Label':<8} {'Status':<10}")
        print("-"*60)
        
        for i in range(min(20, len(sorted_pairs))):
            c_idx, p_idx = sorted_pairs[i]
            score = sorted_scores[i]
            label = sorted_labels[i]
            status = "✓ HIT!" if label == 1 else "✗ Miss"
            print(f"{i+1:<6} {c_idx:<8} {p_idx:<8} {score:<10.4f} {label:<8} {status:<10}")
        
        # CREATE VISUALIZATIONS
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Precision at K plot
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(k_values[:len(precisions)], precisions, 'b-o', linewidth=2, markersize=8)
        ax1.axhline(y=random_precision, color='r', linestyle='--', label=f'Random baseline ({random_precision:.3f})')
        ax1.set_xlabel('Top K edges', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision@K', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xscale('log')
        
        # 2. Recall at K plot
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(k_values[:len(recalls)], recalls, 'g-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Top K edges', fontsize=12)
        ax2.set_ylabel('Recall', fontsize=12)
        ax2.set_title('Recall@K', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # 3. Precision-Recall curve
        ax3 = plt.subplot(2, 3, 3)
        precision_curve, recall_curve, _ = precision_recall_curve(labels_array, scores_array)
        ax3.plot(recall_curve, precision_curve, 'purple', linewidth=2, 
                label=f'AP={avg_precision:.3f}')
        ax3.axhline(y=random_precision, color='r', linestyle='--', alpha=0.5, 
                   label=f'Random ({random_precision:.3f})')
        ax3.set_xlabel('Recall', fontsize=12)
        ax3.set_ylabel('Precision', fontsize=12)
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Score distribution comparison
        ax4 = plt.subplot(2, 3, 4)
        positive_scores = scores_array[labels_array == 1]
        negative_scores = scores_array[labels_array == 0]
        
        bins = np.linspace(scores_array.min(), scores_array.max(), 50)
        ax4.hist(positive_scores, bins=bins, alpha=0.6, label=f'True positives (n={len(positive_scores)})', 
                color='green', density=True)
        ax4.hist(negative_scores, bins=bins, alpha=0.6, label=f'True negatives (n={len(negative_scores)})', 
                color='red', density=True)
        ax4.axvline(np.mean(positive_scores), color='darkgreen', linestyle='--', linewidth=2)
        ax4.axvline(np.mean(negative_scores), color='darkred', linestyle='--', linewidth=2)
        ax4.set_xlabel('Attention Score', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Score Distribution by True Label', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cumulative hits plot
        ax5 = plt.subplot(2, 3, 5)
        cumulative_hits = np.cumsum(sorted_labels)
        x_range = np.arange(1, len(cumulative_hits) + 1)
        ax5.plot(x_range[:1000], cumulative_hits[:1000], 'b-', linewidth=2, label='Actual')
        # Random expectation
        random_hits = x_range * random_precision
        ax5.plot(x_range[:1000], random_hits[:1000], 'r--', linewidth=2, label='Random expectation')
        ax5.set_xlabel('Number of top edges selected', fontsize=12)
        ax5.set_ylabel('Cumulative true positives', fontsize=12)
        ax5.set_title('Cumulative True Positives', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Enrichment plot
        ax6 = plt.subplot(2, 3, 6)
        enrichment = []
        for k in range(10, min(1000, len(sorted_labels)), 10):
            top_k_precision = np.sum(sorted_labels[:k]) / k
            enrichment_factor = top_k_precision / random_precision if random_precision > 0 else 0
            enrichment.append(enrichment_factor)
        
        ax6.plot(range(10, min(1000, len(sorted_labels)), 10), enrichment, 'purple', linewidth=2)
        ax6.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No enrichment')
        ax6.set_xlabel('Top K edges', fontsize=12)
        ax6.set_ylabel('Enrichment Factor', fontsize=12)
        ax6.set_title('Enrichment Over Random', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.suptitle('Soft Edge Validation: Do Selected Edges Match True Labels?', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'soft_edge_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {results_dir}/soft_edge_validation.png")
        
        # Save detailed results
        validation_results = {
            'total_pairs': len(scores_list),
            'positive_pairs': int(np.sum(labels_array)),
            'average_precision': float(avg_precision),
            'random_baseline': float(random_precision),
            'improvement_factor': float(avg_precision/random_precision) if random_precision > 0 else 0,
            'precision_at_k': {k: float(p) for k, p in zip(k_values[:len(precisions)], precisions)},
            'recall_at_k': {k: float(r) for k, r in zip(k_values[:len(recalls)], recalls)},
            'top_20_predictions': [
                {
                    'rank': i+1,
                    'compound_idx': int(sorted_pairs[i][0]),
                    'protein_idx': int(sorted_pairs[i][1]),
                    'score': float(sorted_scores[i]),
                    'is_true_positive': bool(sorted_labels[i] == 1)
                }
                for i in range(min(20, len(sorted_pairs)))
            ]
        }
        
        with open(os.path.join(results_dir, 'soft_edge_validation_results.json'), 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"Detailed results saved to: {results_dir}/soft_edge_validation_results.json")
        
        print("\n" + "="*80)
        print("SOFT EDGE VALIDATION COMPLETE")
        print("="*80)
        
        return validation_results
        
def analyze_meaningful_validation_categories(data, results_dir="results"):
    """Analyze and print meaningful categories for validation sets (1-5% of total CP pairs)"""
    
    print("\n" + "="*80)
    print("ANALYZING MEANINGFUL CATEGORIES FOR SOFT-EDGE VALIDATION")
    print("="*80)
    
    # Extract all positive CP pairs
    positive_pairs = []
    if 'cp_edge_index' in data and data['cp_edge_index'].size > 0:
        cp_edge_list = data['cp_edge_index'].T
        for i in range(0, len(cp_edge_list), 2):  # skip duplicate edges
            c_idx, p_idx = cp_edge_list[i]
            positive_pairs.append((c_idx, p_idx - data['num_compounds']))
    
    total_cp_pairs = len(positive_pairs)
    target_min = int(total_cp_pairs * 0.01)  # 1%
    target_max = int(total_cp_pairs * 0.05)  # 5%
    
    print(f"\nTotal CP pairs: {total_cp_pairs}")
    print(f"Target range: {target_min} - {target_max} pairs (1-5%)")
    print("\n" + "-"*60)
    
    # Load CLINICAL/BIOLOGICAL data
    compound_phases = data.get('compound_phases', {})
    protein_functions = data.get('protein_functions', {})
    interaction_types = data.get('interaction_types', {})
    binding_affinities = data.get('binding_affinities', {})
    
    validation_categories = {}
    
    # ========== CLINICAL/BIOLOGICAL CATEGORIES ==========
    
    # 1. CLINICAL PHASE-BASED CATEGORIES (compound space)
    # Group by clinical development stage
    clinical_phase_groups = {
        'preclinical': [],
        'early_clinical': [],  # Phase 1/2
        'late_clinical': [],   # Phase 3
        'market': []          # Approved/Phase 4
    }
    
    for c_idx, p_idx in positive_pairs:
        phase = compound_phases.get(c_idx, 'Unknown')
        if phase in ['Preclinical', 'Unknown']:
            clinical_phase_groups['preclinical'].append((c_idx, p_idx))
        elif phase in ['Phase 1', 'Phase 2']:
            clinical_phase_groups['early_clinical'].append((c_idx, p_idx))
        elif phase == 'Phase 3':
            clinical_phase_groups['late_clinical'].append((c_idx, p_idx))
        elif phase in ['Approved', 'Phase 4']:
            clinical_phase_groups['market'].append((c_idx, p_idx))
    
    # Add phase-based categories
    validation_categories['preclinical_compounds'] = {
        'pairs': clinical_phase_groups['preclinical'],
        'description': 'Preclinical/experimental compounds',
        'rationale': 'Early-stage compounds with uncertain clinical future'
    }
    
    validation_categories['early_clinical'] = {
        'pairs': clinical_phase_groups['early_clinical'],
        'description': 'Phase 1/2 clinical compounds',
        'rationale': 'Compounds in human testing but not yet proven'
    }
    
    validation_categories['late_clinical'] = {
        'pairs': clinical_phase_groups['late_clinical'],
        'description': 'Phase 3 clinical compounds',
        'rationale': 'Compounds close to approval with high success probability'
    }
    
    validation_categories['marketed_drugs'] = {
        'pairs': clinical_phase_groups['market'],
        'description': 'Approved/marketed drugs',
        'rationale': 'Clinically validated compounds with proven efficacy'
    }
    
    # 2. PROTEIN FUNCTION CATEGORIES (protein space)
    # Group by major therapeutic target classes
    protein_therapeutic_classes = {
        'kinases': [],
        'gpcrs': [],
        'ion_channels': [],
        'nuclear_receptors': [],
        'proteases': [],
        'transporters': [],
        'enzymes_other': [],
        'epigenetic': [],
        'immune': []
    }
    
    for c_idx, p_idx in positive_pairs:
        func = str(protein_functions.get(p_idx, '')).lower()
        
        if 'kinase' in func:
            protein_therapeutic_classes['kinases'].append((c_idx, p_idx))
        elif any(term in func for term in ['gpcr', 'g-protein', 'receptor']):
            protein_therapeutic_classes['gpcrs'].append((c_idx, p_idx))
        elif any(term in func for term in ['channel', 'ion']):
            protein_therapeutic_classes['ion_channels'].append((c_idx, p_idx))
        elif 'nuclear' in func and 'receptor' in func:
            protein_therapeutic_classes['nuclear_receptors'].append((c_idx, p_idx))
        elif 'protease' in func:
            protein_therapeutic_classes['proteases'].append((c_idx, p_idx))
        elif 'transport' in func:
            protein_therapeutic_classes['transporters'].append((c_idx, p_idx))
        elif any(term in func for term in ['histone', 'methyltransferase', 'deacetylase', 'epigenetic']):
            protein_therapeutic_classes['epigenetic'].append((c_idx, p_idx))
        elif any(term in func for term in ['immune', 'cytokine', 'interleukin', 'checkpoint']):
            protein_therapeutic_classes['immune'].append((c_idx, p_idx))
        elif 'enzyme' in func or 'ase' in func:  # General enzymes
            protein_therapeutic_classes['enzymes_other'].append((c_idx, p_idx))
    
    # Add protein class categories
    for class_name, pairs in protein_therapeutic_classes.items():
        validation_categories[f'{class_name}_targets'] = {
            'pairs': pairs,
            'description': f'{class_name.replace("_", " ").title()} protein targets',
            'rationale': f'Major therapeutic target class in drug discovery'
        }
    
    # 3. CLINICAL EFFICACY CATEGORIES (based on binding affinity)
    if binding_affinities:
        efficacy_groups = {
            'high_potency': [],      # pChEMBL > 8 (nM range)
            'moderate_potency': [],  # pChEMBL 6-8 (nM to μM)
            'low_potency': [],       # pChEMBL < 6 (μM range)
        }
        
        for c_idx, p_idx in positive_pairs:
            affinity = binding_affinities.get((c_idx, p_idx), 0)
            if affinity > 8:
                efficacy_groups['high_potency'].append((c_idx, p_idx))
            elif 6 <= affinity <= 8:
                efficacy_groups['moderate_potency'].append((c_idx, p_idx))
            elif affinity > 0 and affinity < 6:
                efficacy_groups['low_potency'].append((c_idx, p_idx))
        
        validation_categories['high_potency_drugs'] = {
            'pairs': efficacy_groups['high_potency'],
            'description': 'High potency interactions (pChEMBL > 8)',
            'rationale': 'Clinically relevant nanomolar potency'
        }
        
        validation_categories['moderate_potency_drugs'] = {
            'pairs': efficacy_groups['moderate_potency'],
            'description': 'Moderate potency (pChEMBL 6-8)',
            'rationale': 'Typical drug-like potency range'
        }
    
    # 4. CLINICAL DEVELOPMENT RISK CATEGORIES
    # Combine phase and target info for risk assessment
    from collections import defaultdict
    protein_compound_count = defaultdict(int)
    for c_idx, p_idx in positive_pairs:
        protein_compound_count[p_idx] += 1
    
    risk_categories = {
        'high_confidence': [],    # Approved drugs on well-validated targets
        'exploratory': [],        # Early compounds on novel targets
        'repurposing': []        # Approved drugs on new targets
    }
    
    for c_idx, p_idx in positive_pairs:
        phase = compound_phases.get(c_idx, 'Unknown')
        target_validation = protein_compound_count[p_idx]
        
        if phase in ['Approved', 'Phase 4'] and target_validation > 10:
            risk_categories['high_confidence'].append((c_idx, p_idx))
        elif phase in ['Preclinical', 'Phase 1'] and target_validation < 5:
            risk_categories['exploratory'].append((c_idx, p_idx))
        elif phase in ['Approved', 'Phase 4'] and target_validation < 5:
            risk_categories['repurposing'].append((c_idx, p_idx))
    
    validation_categories['high_confidence_clinical'] = {
        'pairs': risk_categories['high_confidence'],
        'description': 'Approved drugs on well-validated targets',
        'rationale': 'Lowest clinical risk - proven drug-target pairs'
    }
    
    validation_categories['exploratory_clinical'] = {
        'pairs': risk_categories['exploratory'],
        'description': 'Early compounds on novel targets',
        'rationale': 'High risk/high reward - new chemical/biological space'
    }
    
    validation_categories['drug_repurposing'] = {
        'pairs': risk_categories['repurposing'],
        'description': 'Approved drugs tested on new targets',
        'rationale': 'Drug repurposing opportunities'
    }
    
    # 5. DISEASE AREA CATEGORIES (inferred from protein function)
    disease_areas = {
        'oncology': [],
        'neurology': [],
        'immunology': [],
        'metabolic': [],
        'cardiovascular': [],
        'infectious': []
    }
    
    for c_idx, p_idx in positive_pairs:
        func = str(protein_functions.get(p_idx, '')).lower()
        
        if any(term in func for term in ['tumor', 'cancer', 'onco', 'proliferation', 'apoptosis']):
            disease_areas['oncology'].append((c_idx, p_idx))
        elif any(term in func for term in ['neuro', 'brain', 'dopamine', 'serotonin', 'gaba']):
            disease_areas['neurology'].append((c_idx, p_idx))
        elif any(term in func for term in ['immune', 'cytokine', 'interleukin', 'inflammation']):
            disease_areas['immunology'].append((c_idx, p_idx))
        elif any(term in func for term in ['metabol', 'insulin', 'glucose', 'lipid']):
            disease_areas['metabolic'].append((c_idx, p_idx))
        elif any(term in func for term in ['cardiac', 'heart', 'vascular', 'blood']):
            disease_areas['cardiovascular'].append((c_idx, p_idx))
        elif any(term in func for term in ['viral', 'bacterial', 'pathogen', 'antibiotic']):
            disease_areas['infectious'].append((c_idx, p_idx))
    
    for area, pairs in disease_areas.items():
        if pairs:  # Only add if we have pairs
            validation_categories[f'{area}_drugs'] = {
                'pairs': pairs,
                'description': f'{area.title()} therapeutic area',
                'rationale': f'Disease-specific drug discovery in {area}'
            }
    
    # 6. CLINICAL FAILURE RISK (orphan drugs and rare targets)
    # These are clinically important but challenging
    compound_protein_count = defaultdict(int)
    for c_idx, p_idx in positive_pairs:
        compound_protein_count[c_idx] += 1
    
    clinical_challenge_pairs = []
    for c_idx, p_idx in positive_pairs:
        # Orphan drug criteria: targets rare diseases or has limited target profile
        if (protein_compound_count[p_idx] <= 3 or  # Rare target
            compound_protein_count[c_idx] == 1):    # Highly specific compound
            clinical_challenge_pairs.append((c_idx, p_idx))
    
    validation_categories['orphan_drug_space'] = {
        'pairs': clinical_challenge_pairs,
        'description': 'Orphan drugs and rare disease targets',
        'rationale': 'Clinically important but data-sparse interactions'
    }
    
    # 7. COMBINATION THERAPY CANDIDATES
    # Compounds/proteins involved in multiple interactions (polypharmacology)
    combination_pairs = []
    for c_idx, p_idx in positive_pairs:
        if compound_protein_count[c_idx] >= 5:  # Promiscuous compounds
            combination_pairs.append((c_idx, p_idx))
    
    validation_categories['combination_therapy'] = {
        'pairs': combination_pairs,
        'description': 'Multi-target compounds for combination therapy',
        'rationale': 'Polypharmacology for complex diseases'
    }
    
    # Rest of the function remains the same (printing, analysis, etc.)
    # Print analysis of each category
    print("\nVALIDATION CATEGORY ANALYSIS:")
    print("="*80)
    
    suitable_categories = []
    all_categories_summary = []
    
    for cat_name, cat_data in validation_categories.items():
        num_pairs = len(cat_data['pairs'])
        percentage = (num_pairs / total_cp_pairs * 100) if total_cp_pairs > 0 else 0
        
        print(f"\n{cat_name.upper()}:")
        print(f"  Description: {cat_data['description']}")
        print(f"  Rationale: {cat_data['rationale']}")
        print(f"  Size: {num_pairs} pairs ({percentage:.2f}%)")
        
        # Store summary for table
        all_categories_summary.append({
            'Category': cat_name,
            'Count': num_pairs,
            'Percentage': percentage,
            'Status': ''
        })
        
        if target_min <= num_pairs <= target_max:
            print(f"  ✓ SUITABLE FOR VALIDATION (within 1-5% range)")
            suitable_categories.append((cat_name, cat_data, num_pairs, percentage))
            all_categories_summary[-1]['Status'] = '✓ Suitable'
        elif num_pairs < target_min:
            print(f"  ✗ Too small ({percentage:.2f}% < 1%)")
            all_categories_summary[-1]['Status'] = 'Too small'
        else:
            print(f"  ✗ Too large ({percentage:.2f}% > 5%)")
            all_categories_summary[-1]['Status'] = 'Too large'
        
        # Sample some pairs for inspection
        if num_pairs > 0:
            sample_size = min(5, num_pairs)
            sample_pairs = cat_data['pairs'][:sample_size]
            print(f"  Sample pairs: {sample_pairs}")
    
    # Rest of function continues...
    # Print summary table
    print("\n" + "="*80)
    print("CATEGORY SUMMARY TABLE:")
    print("="*80)
    print(f"{'Category':<25} {'Count':>8} {'Percentage':>12} {'Status':<15}")
    print("-"*65)
    
    # Sort by percentage for better overview
    all_categories_summary.sort(key=lambda x: x['Percentage'], reverse=True)
    
    for cat in all_categories_summary:
        print(f"{cat['Category']:<25} {cat['Count']:>8} {cat['Percentage']:>11.2f}% {cat['Status']:<15}")
    
    # Create CLINICAL validation sets
    print("\n" + "-"*60)
    print("CREATING CLINICAL VALIDATION SETS:")
    print("-"*60)
    
    # 1. CLINICAL PROGRESSION SET
    clinical_progression_pairs = set()
    
    # Add some early clinical
    if 'early_clinical' in validation_categories:
        early_sample = validation_categories['early_clinical']['pairs'][:target_max//3]
        clinical_progression_pairs.update(early_sample)
    
    # Add some late clinical
    if 'late_clinical' in validation_categories:
        late_sample = validation_categories['late_clinical']['pairs'][:target_max//3]
        clinical_progression_pairs.update(late_sample)
    
    # Add some marketed
    if 'marketed_drugs' in validation_categories:
        market_sample = validation_categories['marketed_drugs']['pairs'][:target_max//3]
        clinical_progression_pairs.update(market_sample)
    
    print(f"\n1. CLINICAL PROGRESSION SET: {len(clinical_progression_pairs)} pairs ({len(clinical_progression_pairs)/total_cp_pairs*100:.2f}%)")
    print("   Spans drug development pipeline: early clinical → late clinical → market")
    
    # 2. THERAPEUTIC DIVERSITY SET
    therapeutic_diversity_pairs = set()
    
    # Sample from each therapeutic area
    for area in ['oncology_drugs', 'neurology_drugs', 'immunology_drugs', 'metabolic_drugs']:
        if area in validation_categories and validation_categories[area]['pairs']:
            area_sample = validation_categories[area]['pairs'][:target_max//4]
            therapeutic_diversity_pairs.update(area_sample)
    
    print(f"\n2. THERAPEUTIC DIVERSITY SET: {len(therapeutic_diversity_pairs)} pairs ({len(therapeutic_diversity_pairs)/total_cp_pairs*100:.2f}%)")
    print("   Covers multiple disease areas for broad clinical applicability")
    
    # 3. TARGET CLASS DIVERSITY SET
    target_diversity_pairs = set()
    
    # Sample from major target classes
    for target_class in ['kinases_targets', 'gpcrs_targets', 'ion_channels_targets', 'proteases_targets']:
        if target_class in validation_categories and validation_categories[target_class]['pairs']:
            class_sample = validation_categories[target_class]['pairs'][:target_max//4]
            target_diversity_pairs.update(class_sample)
    
    print(f"\n3. TARGET CLASS DIVERSITY SET: {len(target_diversity_pairs)} pairs ({len(target_diversity_pairs)/total_cp_pairs*100:.2f}%)")
    print("   Major druggable target families")
    
    # Convert pairs to regular Python types before saving
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return tuple(convert_to_python_types(item) for item in obj)
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, set):
            return list(convert_to_python_types(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(val) for key, val in obj.items()}
        else:
            return obj
    
    # Save validation sets
    validation_sets = {
        'total_cp_pairs': int(total_cp_pairs),
        'categories': {name: {'size': int(len(data['pairs'])), 
                              'percentage': float(len(data['pairs'])/total_cp_pairs*100) if total_cp_pairs > 0 else 0.0,
                              'description': data['description']}
                      for name, data in validation_categories.items()},
        'clinical_progression_set': convert_to_python_types(list(clinical_progression_pairs)),
        'therapeutic_diversity_set': convert_to_python_types(list(therapeutic_diversity_pairs)),
        'target_diversity_set': convert_to_python_types(list(target_diversity_pairs))
    }
    
    import json
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'validation_set_analysis.json'), 'w') as f:
        json.dump(validation_sets, f, indent=2)
    
    print(f"\n\nValidation set analysis saved to: {os.path.join(results_dir, 'validation_set_analysis.json')}")

    # RECOMMENDATION ENGINE
    print("\n" + "="*80)
    print("CLINICAL VALIDATION SET RECOMMENDATIONS:")
    print("="*80)
    
    print("\nFor validating soft-edge masking with clinical relevance:")
    print("\n1. **CLINICAL PROGRESSION SET** - Tests generalization across development stages")
    print("2. **HIGH CONFIDENCE CLINICAL** - Well-validated drug-target pairs")
    print("3. **EXPLORATORY CLINICAL** - Novel chemical/biological space")
    print("4. **ORPHAN DRUG SPACE** - Rare diseases with limited data")
    print("5. **TARGET CLASS SPECIFIC** - e.g., kinases_targets for focused validation")
    print("\nThese sets test whether soft-edge masking captures clinically meaningful")
    print("drug-target relationships beyond simple chemical similarity.")
    
    return validation_categories, suitable_categories

def analyze_misclassified_interactions(y_true, y_pred, test_pairs, data, results_dir="results"):
    """Analyze misclassified true positive compounds, proteins, and pairs (false negatives)"""
    
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import textwrap
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ANALYSIS OF MISCLASSIFIED TRUE POSITIVE INTERACTIONS (FALSE NEGATIVES)")
    print("="*80)
    
    # Identify false negatives (true positives that were predicted as negative)
    false_negatives = []
    false_negatives_indices = []
    
    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label == 1 and pred_label == 0:  # False negative
            if i < len(test_pairs):  # It's from the positive test pairs
                false_negatives.append(test_pairs[i])
                false_negatives_indices.append(i)
    
    if len(false_negatives) == 0:
        print("No false negatives found - all true positive interactions were correctly predicted!")
        return {}
    
    print(f"\nFound {len(false_negatives)} false negatives out of {len(test_pairs)} positive test pairs")
    print(f"False negative rate: {len(false_negatives)/len(test_pairs)*100:.2f}%")
    
    # Load additional data if available
    compound_phases = data.get('compound_phases', {})
    protein_functions = data.get('protein_functions', {})
    binding_affinities = data.get('binding_affinities', {})
    compound_names = data.get('compound_names', {})
    protein_names = data.get('protein_names', {})
    
    # Count misclassifications per compound and protein
    compound_fn_counts = {}
    protein_fn_counts = {}
    
    for c_idx, p_idx in false_negatives:
        compound_fn_counts[c_idx] = compound_fn_counts.get(c_idx, 0) + 1
        protein_fn_counts[p_idx] = protein_fn_counts.get(p_idx, 0) + 1
    
    print("\n" + "-"*60)
    print("TOP MISCLASSIFIED COMPOUNDS (False Negatives)")
    print("-"*60)
    
    # Sort compounds by misclassification count
    top_compounds = sorted(compound_fn_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print(f"\nTop 20 compounds with most false negatives:")
    print(f"{'Rank':<6} {'Compound ID':<15} {'Compound Name':<35} {'Phase':<15} {'FN Count':<10} {'% of FNs':<10}")
    print("-"*110)
    
    for rank, (c_idx, count) in enumerate(top_compounds, 1):
        compound_name = compound_names.get(c_idx, f"Compound_{c_idx}")[:35]
        phase = compound_phases.get(c_idx, "Unknown")
        percentage = count / len(false_negatives) * 100
        print(f"{rank:<6} {f'C{c_idx}':<15} {compound_name:<35} {phase:<15} {count:<10} {percentage:<10.1f}%")
    
    print("\n" + "-"*60)
    print("TOP MISCLASSIFIED PROTEINS (False Negatives)")
    print("-"*60)
    
    # Sort proteins by misclassification count
    top_proteins = sorted(protein_fn_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print(f"\nTop 20 proteins with most false negatives:")
    print(f"{'Rank':<6} {'Protein ID':<15} {'Protein Name':<35} {'Function':<30} {'FN Count':<10} {'% of FNs':<10}")
    print("-"*125)
    
    for rank, (p_idx, count) in enumerate(top_proteins, 1):
        protein_name = protein_names.get(p_idx, f"Protein_{p_idx}")[:35]
        protein_func = str(protein_functions.get(p_idx, "Unknown"))[:30]
        percentage = count / len(false_negatives) * 100
        print(f"{rank:<6} {f'P{p_idx}':<15} {protein_name:<35} {protein_func:<30} {count:<10} {percentage:<10.1f}%")
    
    print("\n" + "-"*60)
    print("SPECIFIC MISCLASSIFIED PAIRS (Top 50 False Negatives)")
    print("-"*60)
    
    # Create detailed list of ALL misclassified pairs
    all_pair_details = []
    for c_idx, p_idx in false_negatives:
        compound_name = compound_names.get(c_idx, f"Compound_{c_idx}")
        protein_name = protein_names.get(p_idx, f"Protein_{p_idx}")
        phase = compound_phases.get(c_idx, "Unknown")
        function = protein_functions.get(p_idx, "Unknown")
        affinity = binding_affinities.get((c_idx, p_idx), None)
        
        all_pair_details.append({
            'c_idx': c_idx,
            'p_idx': p_idx,
            'compound_name': compound_name,
            'protein_name': protein_name,
            'phase': phase,
            'function': function,
            'affinity': affinity if affinity is not None else -1,
            'pair_string': f"C{c_idx}-P{p_idx}"
        })
    
    # Sort by affinity (highest first) to show most potent misclassified interactions
    all_pair_details.sort(key=lambda x: x['affinity'], reverse=True)
    
    print(f"\nTop 50 misclassified compound-protein pairs (sorted by binding affinity):")
    print(f"{'Rank':<6} {'Pair':<15} {'Compound':<30} {'Protein':<30} {'Phase':<12} {'pChEMBL':<10}")
    print("-"*130)
    
    for rank, pair in enumerate(all_pair_details[:50], 1):
        affinity_str = f"{pair['affinity']:.2f}" if pair['affinity'] > 0 else "N/A"
        print(f"{rank:<6} {pair['pair_string']:<15} {pair['compound_name'][:30]:<30} "
              f"{pair['protein_name'][:30]:<30} {pair['phase']:<12} {affinity_str:<10}")
    
    # CREATE ENHANCED VISUALIZATIONS WITH COMPLETE LISTS
    fig = plt.figure(figsize=(20, 24))
    
    # Create a 3x2 grid layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2], hspace=0.3, wspace=0.2)
    
    # PLOT 1: Top 10 Misclassified Compounds
    ax1 = fig.add_subplot(gs[0, 0])
    if len(top_compounds) > 0:
        # Take only top 10
        top_10_compounds = top_compounds[:10]
        compound_labels = []
        compound_counts = []
        
        for c_idx, count in top_10_compounds:
            # Get compound name, truncate if too long
            c_name = compound_names.get(c_idx, f"C{c_idx}")
            if len(c_name) > 25:
                c_name = c_name[:22] + "..."
            compound_labels.append(c_name)
            compound_counts.append(count)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(compound_labels))
        ax1.barh(y_pos, compound_counts, color='coral', edgecolor='darkred', linewidth=1.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(compound_labels, fontsize=10)
        ax1.set_xlabel('Number of False Negatives', fontsize=11)
        ax1.invert_yaxis()  # Top compound at the top
        
        # Add value labels
        for i, (count, label) in enumerate(zip(compound_counts, compound_labels)):
            ax1.text(count + 0.1, i, str(count), ha='left', va='center', fontsize=10)
        
        ax1.grid(True, alpha=0.3, axis='x')
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
    
    # PLOT 2: Top 10 Misclassified Proteins
    ax2 = fig.add_subplot(gs[0, 1])
    if len(top_proteins) > 0:
        # Take only top 10
        top_10_proteins = top_proteins[:10]
        protein_labels = []
        protein_counts = []
        
        for p_idx, count in top_10_proteins:
            # Get protein name, truncate if too long
            p_name = protein_names.get(p_idx, f"P{p_idx}")
            if len(p_name) > 25:
                p_name = p_name[:22] + "..."
            protein_labels.append(p_name)
            protein_counts.append(count)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(protein_labels))
        ax2.barh(y_pos, protein_counts, color='darkgreen', edgecolor='black', linewidth=1.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(protein_labels, fontsize=10)
        ax2.set_xlabel('Number of False Negatives', fontsize=11)
        ax2.invert_yaxis()  # Top protein at the top
        
        # Add value labels
        for i, (count, label) in enumerate(zip(protein_counts, protein_labels)):
            ax2.text(count + 0.1, i, str(count), ha='left', va='center', fontsize=10)
        
        ax2.grid(True, alpha=0.3, axis='x')
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
    
    # PLOT 3: Distribution of Misclassifications (per entity)
    ax3 = fig.add_subplot(gs[1, 0])
    compound_counts_list = list(compound_fn_counts.values())
    protein_counts_list = list(protein_fn_counts.values())
    
    if compound_counts_list or protein_counts_list:
        # Create bins for the histogram
        max_count = max(max(compound_counts_list) if compound_counts_list else 0,
                       max(protein_counts_list) if protein_counts_list else 0)
        bins = np.arange(0, max_count + 2) - 0.5
        
        # Plot histograms
        if compound_counts_list:
            ax3.hist(compound_counts_list, bins=bins, alpha=0.6, label='Compounds', 
                    color='coral', edgecolor='darkred', linewidth=1)
        if protein_counts_list:
            ax3.hist(protein_counts_list, bins=bins, alpha=0.6, label='Proteins', 
                    color='darkgreen', edgecolor='black', linewidth=1)
        
        ax3.set_xlabel('Number of Misclassifications per Entity', fontsize=11)
        ax3.set_ylabel('Frequency (Number of Entities)', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        if compound_counts_list:
            comp_mean = np.mean(compound_counts_list)
            comp_median = np.median(compound_counts_list)
            ax3.axvline(comp_mean, color='coral', linestyle='--', alpha=0.7, linewidth=2)
        if protein_counts_list:
            prot_mean = np.mean(protein_counts_list)
            prot_median = np.median(protein_counts_list)
            ax3.axvline(prot_mean, color='darkgreen', linestyle='--', alpha=0.7, linewidth=2)
    else:
        ax3.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax3.transAxes)
    
    # PLOT 4: Connectivity Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    # Compare the connectivity (degree) of misclassified vs correctly classified entities
    all_compound_connections = {}
    all_protein_connections = {}
    
    # Count all connections for each compound and protein
    for c_idx, p_idx in test_pairs:
        all_compound_connections[c_idx] = all_compound_connections.get(c_idx, 0) + 1
        all_protein_connections[p_idx] = all_protein_connections.get(p_idx, 0) + 1
    
    # Get connectivity for misclassified vs correctly classified
    misclassified_compound_degrees = [all_compound_connections.get(c, 0) for c in compound_fn_counts.keys()]
    misclassified_protein_degrees = [all_protein_connections.get(p, 0) for p in protein_fn_counts.keys()]
    
    # Get correctly classified entities
    correctly_classified_compounds = set(all_compound_connections.keys()) - set(compound_fn_counts.keys())
    correctly_classified_proteins = set(all_protein_connections.keys()) - set(protein_fn_counts.keys())
    
    correct_compound_degrees = [all_compound_connections[c] for c in correctly_classified_compounds]
    correct_protein_degrees = [all_protein_connections[p] for p in correctly_classified_proteins]
    
    # Create box plots for connectivity comparison
    data_to_plot = []
    labels_to_plot = []
    
    if misclassified_compound_degrees:
        data_to_plot.append(misclassified_compound_degrees)
        labels_to_plot.append('Misclass.\nCompounds')
    if correct_compound_degrees:
        data_to_plot.append(correct_compound_degrees)
        labels_to_plot.append('Correct\nCompounds')
    if misclassified_protein_degrees:
        data_to_plot.append(misclassified_protein_degrees)
        labels_to_plot.append('Misclass.\nProteins')
    if correct_protein_degrees:
        data_to_plot.append(correct_protein_degrees)
        labels_to_plot.append('Correct\nProteins')
    
    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                         medianprops=dict(color='red', linewidth=2),
                         boxprops=dict(facecolor='lightblue', alpha=0.7))
        
        # Color the boxes
        colors = ['coral', 'lightcoral', 'darkgreen', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax4.set_ylabel('Degree (Number of Connections)', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        for i, data in enumerate(data_to_plot):
            if data:
                mean_val = np.mean(data)
                ax4.plot(i+1, mean_val, 'D', color='blue', markersize=8, label='Mean' if i == 0 else '')
        
        if any(data_to_plot):
            ax4.legend(loc='upper right', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
    
    # PLOT 5 & 6: COMPLETE LIST OF ALL MISCLASSIFIED PAIRS (spanning bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create text listing of ALL misclassified pairs
    # Create formatted text for all pairs
    pairs_per_column = 30  # Number of pairs per column
    num_columns = min(6, (len(all_pair_details) + pairs_per_column - 1) // pairs_per_column)
    column_width = 1.0 / num_columns
    
    # Header
    header = "No.  Pair         Compound                    Protein                     Affinity"
    header_line = "—" * 80
    
    for col in range(num_columns):
        x_position = col * column_width + 0.01
        y_position = 0.93
        
        # Add header for each column
        ax5.text(x_position, y_position, header, transform=ax5.transAxes,
                fontsize=8, fontweight='bold', fontfamily='monospace', va='top')
        ax5.text(x_position, y_position - 0.02, header_line, transform=ax5.transAxes,
                fontsize=8, fontfamily='monospace', va='top')
        
        # Add pairs for this column
        start_idx = col * pairs_per_column
        end_idx = min(start_idx + pairs_per_column, len(all_pair_details))
        
        pair_texts = []
        for i in range(start_idx, end_idx):
            pair = all_pair_details[i]
            # Format each line
            compound_name = pair['compound_name'][:25]
            protein_name = pair['protein_name'][:25]
            affinity_str = f"{pair['affinity']:.2f}" if pair['affinity'] > 0 else "N/A"
            
            line = f"{i+1:3d}  {pair['pair_string']:<11} {compound_name:<27} {protein_name:<27} {affinity_str:>8}"
            pair_texts.append(line)
        
        # Join all lines and add to plot
        column_text = '\n'.join(pair_texts)
        ax5.text(x_position, y_position - 0.04, column_text, transform=ax5.transAxes,
                fontsize=7, fontfamily='monospace', va='top')
    
    # Add summary statistics at the bottom
    summary_text = (f"\nTotal False Negatives: {len(all_pair_details)} | "
                   f"False Negative Rate: {len(all_pair_details)/len(test_pairs)*100:.1f}% | "
                   f"Unique Compounds: {len(compound_fn_counts)} | "
                   f"Unique Proteins: {len(protein_fn_counts)}")
    
    ax5.text(0.5, 0.02, summary_text, transform=ax5.transAxes,
            fontsize=10, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('False Negative Analysis - Complete Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # SAVE THE PLOT
    plot_filepath = os.path.join(results_dir, 'false_negative_analysis_complete.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Complete analysis plot saved to: {plot_filepath}")
    
    # Also create a separate detailed text file with ALL pairs
    text_filepath = os.path.join(results_dir, 'all_misclassified_pairs.txt')
    with open(text_filepath, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"COMPLETE LIST OF ALL {len(all_pair_details)} MISCLASSIFIED PAIRS (FALSE NEGATIVES)\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Rank':<6} {'Pair':<15} {'Compound':<35} {'Protein':<35} {'Phase':<12} {'pChEMBL':<10}\n")
        f.write("-"*130 + "\n")
        
        for rank, pair in enumerate(all_pair_details, 1):
            affinity_str = f"{pair['affinity']:.2f}" if pair['affinity'] > 0 else "N/A"
            f.write(f"{rank:<6} {pair['pair_string']:<15} {pair['compound_name'][:35]:<35} "
                   f"{pair['protein_name'][:35]:<35} {pair['phase']:<12} {affinity_str:<10}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*100 + "\n")
        f.write(f"Total False Negatives: {len(all_pair_details)}\n")
        f.write(f"False Negative Rate: {len(all_pair_details)/len(test_pairs)*100:.2f}%\n")
        f.write(f"Unique Misclassified Compounds: {len(compound_fn_counts)}\n")
        f.write(f"Unique Misclassified Proteins: {len(protein_fn_counts)}\n")
    
    print(f"✓ Complete text list saved to: {text_filepath}")
    
    # Save detailed report
    fn_report = {
        'total_false_negatives': len(false_negatives),
        'false_negative_rate': len(false_negatives) / len(test_pairs),
        'top_misclassified_compounds': [(int(c), int(count)) for c, count in top_compounds[:20]],
        'top_misclassified_proteins': [(int(p), int(count)) for p, count in top_proteins[:20]],
        'all_misclassified_pairs': [
            {
                'compound_id': int(pair['c_idx']),
                'protein_id': int(pair['p_idx']),
                'compound_name': pair['compound_name'],
                'protein_name': pair['protein_name'],
                'phase': pair['phase'],
                'affinity': float(pair['affinity']) if pair['affinity'] > 0 else None
            }
            for pair in all_pair_details
        ],
    }
    
    # Save report
    report_filepath = os.path.join(results_dir, 'false_negative_detailed_report.json')
    with open(report_filepath, 'w') as f:
        json.dump(fn_report, f, indent=2)
    print(f"✓ Report saved to: {report_filepath}")
    
    # Save CSV files for easy analysis
    # Top compounds CSV
    csv_filepath = os.path.join(results_dir, 'top_misclassified_compounds.csv')
    with open(csv_filepath, 'w') as f:
        f.write('Rank,Compound_ID,Compound_Name,Clinical_Phase,FN_Count,Percentage\n')
        for rank, (c_idx, count) in enumerate(top_compounds[:20], 1):
            compound_name = compound_names.get(c_idx, f"Compound_{c_idx}").replace(',', ';')
            phase = compound_phases.get(c_idx, "Unknown")
            percentage = count / len(false_negatives) * 100
            f.write(f'{rank},{c_idx},{compound_name},{phase},{count},{percentage:.2f}\n')
    print(f"✓ CSV saved to: {csv_filepath}")
    
    # Top proteins CSV
    csv_filepath = os.path.join(results_dir, 'top_misclassified_proteins.csv')
    with open(csv_filepath, 'w') as f:
        f.write('Rank,Protein_ID,Protein_Name,Function,FN_Count,Percentage\n')
        for rank, (p_idx, count) in enumerate(top_proteins[:20], 1):
            protein_name = protein_names.get(p_idx, f"Protein_{p_idx}").replace(',', ';')
            function = str(protein_functions.get(p_idx, "Unknown")).replace(',', ';')
            percentage = count / len(false_negatives) * 100
            f.write(f'{rank},{p_idx},{protein_name},{function},{count},{percentage:.2f}\n')
    print(f"✓ CSV saved to: {csv_filepath}")
    
    # ALL misclassified pairs CSV
    csv_filepath = os.path.join(results_dir, 'all_misclassified_pairs.csv')
    with open(csv_filepath, 'w') as f:
        f.write('Rank,Compound_ID,Protein_ID,Compound_Name,Protein_Name,Clinical_Phase,pChEMBL\n')
        for rank, pair in enumerate(all_pair_details, 1):
            compound_name = pair['compound_name'].replace(',', ';')
            protein_name = pair['protein_name'].replace(',', ';')
            affinity_str = f"{pair['affinity']:.2f}" if pair['affinity'] > 0 else "N/A"
            f.write(f"{rank},{pair['c_idx']},{pair['p_idx']},{compound_name},{protein_name},{pair['phase']},{affinity_str}\n")
    print(f"✓ Complete pairs CSV saved to: {csv_filepath}")
    
    print("\n" + "="*80)
    print("FALSE NEGATIVE ANALYSIS COMPLETE")
    print("="*80)
    print(f"All files saved to: {os.path.abspath(results_dir)}/")
    print(f"\nKey outputs:")
    print(f"  - false_negative_analysis_complete.png (visualization with complete pair list)")
    print(f"  - all_misclassified_pairs.txt (detailed text file)")
    print(f"  - all_misclassified_pairs.csv (complete CSV for analysis)")
    
    return fn_report

def run_multiple_experiments(num_experiments=30, start_seed=42, results_base_dir="multi_seed_results"):
    """Run multiple experiments with different seeds and aggregate results"""
    
    # Create base directory if it doesn't exist
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Check for existing results in the base directory
    all_existing_seeds = set()
    existing_results = []
    
    # Look for seed_* directories directly in results_base_dir
    if os.path.exists(results_base_dir):
        for item in os.listdir(results_base_dir):
            item_path = os.path.join(results_base_dir, item)
            if item.startswith('seed_') and os.path.isdir(item_path):
                try:
                    seed = int(item.split('_')[1])
                    # Check if this seed completed successfully
                    test_results_path = os.path.join(item_path, 'test_results.json')
                    if os.path.exists(test_results_path):
                        with open(test_results_path, 'r') as f:
                            result = json.load(f)
                            result['seed'] = seed
                            result['status'] = 'success'
                            all_existing_seeds.add(seed)
                            existing_results.append(result)
                except:
                    pass
    
    print(f"\n{'='*80}")
    print(f"MULTI-SEED EXPERIMENT RUNNER")
    print(f"{'='*80}")
    print(f"Found {len(all_existing_seeds)} already completed seeds: {sorted(list(all_existing_seeds))[:10]}{'...' if len(all_existing_seeds) > 10 else ''}")
    print(f"Requested: {num_experiments} NEW experiments")
    
    # Find num_experiments NEW seeds that haven't been run yet
    new_seeds = []
    candidate_seed = start_seed
    
    while len(new_seeds) < num_experiments:
        if candidate_seed not in all_existing_seeds:
            new_seeds.append(candidate_seed)
        candidate_seed += 1
    
    print(f"Will run {len(new_seeds)} new seeds: {new_seeds[:5]}{'...' if len(new_seeds) > 5 else ''}")
    print(f"Seeds range: {new_seeds[0]} to {new_seeds[-1]}")
    print(f"Results will be saved directly to: {results_base_dir}")
    print(f"{'='*80}\n")
    
    # Save experiment configuration
    experiment_config = {
        'num_new_experiments': num_experiments,
        'new_seeds': new_seeds,
        'previously_completed_seeds': sorted(list(all_existing_seeds)),
        'total_seeds_after_completion': len(all_existing_seeds) + len(new_seeds),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(os.path.join(results_base_dir, 'experiment_config.json'), 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    # Storage for new results
    new_results = []
    successful_runs = 0
    failed_runs = 0
    
    # Run new experiments
    for i, seed in enumerate(new_seeds):
        experiment_num = i + 1
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {experiment_num}/{num_experiments} - SEED: {seed}")
        print(f"Progress: {experiment_num/num_experiments*100:.1f}% of new experiments")
        print(f"{'='*60}\n")
        
        # Create seed-specific directory directly in results_base_dir
        seed_dir = os.path.join(results_base_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        
        try:
            # Run single experiment with this seed
            start_time = time.time()
            result = run_single_experiment_with_seed(seed, seed_dir)
            elapsed_time = time.time() - start_time
            
            result['elapsed_time'] = elapsed_time
            result['seed'] = seed
            result['status'] = 'success'
            
            new_results.append(result)
            existing_results.append(result)  # Add to all results
            successful_runs += 1
            
            print(f"\nExperiment {experiment_num} completed successfully in {elapsed_time/60:.2f} minutes")
            print(f"Test F1-Score: {result['test_f1']:.4f}")
            
        except Exception as e:
            print(f"\nERROR in experiment {experiment_num} (seed {seed}): {str(e)}")
            failed_runs += 1
            
            # Save error information
            error_info = {
                'seed': seed,
                'status': 'failed',
                'error': str(e)
            }
            
            with open(os.path.join(seed_dir, 'error.json'), 'w') as f:
                json.dump(error_info, f, indent=2)
            
            new_results.append(error_info)
        
        # Save intermediate results directly in base directory
        save_intermediate_results(new_results, results_base_dir)
        
        # Clear GPU cache after each experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"NEW EXPERIMENTS COMPLETED")
    print(f"Successful new runs: {successful_runs}/{num_experiments}")
    print(f"Failed new runs: {failed_runs}")
    print(f"Total completed seeds (all time): {len(all_existing_seeds) + successful_runs}")
    print(f"{'='*80}\n")
    
    # Aggregate results from ALL successful runs (existing + new)
    if len(existing_results) > 0:
        # Create aggregated results directly in base directory
        aggregated_results = aggregate_experiment_results(existing_results, results_base_dir)
        create_multi_seed_visualizations(aggregated_results, results_base_dir)
        create_comprehensive_analysis(existing_results, results_base_dir)
        print_aggregated_results(aggregated_results)
        
        # Save master summary
        summary = {
            'total_experiments': len(existing_results),
            'seed_range': [min(r['seed'] for r in existing_results), max(r['seed'] for r in existing_results)],
            'aggregated_metrics': aggregated_results,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open(os.path.join(results_base_dir, 'master_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAll results saved to: {results_base_dir}/")
    else:
        print("No successful runs to aggregate!")
    
    return new_results, results_base_dir

def run_single_experiment_with_seed(seed, results_dir):
    """Run a single experiment with a specific seed"""
    
    # Set the global seed
    set_seed(seed)
    
    # Load data
    with open('gnn_graph_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Initialize model
    input_dim = data['node_features'].shape[1]
    hidden_dim = 128
    output_dim = 64
    num_layers = 3
    edge_dim = data.get('edge_attr', np.array([[]])).shape[1] if 'edge_attr' in data and data['edge_attr'].size > 0 else 1
    
    model = DrugTargetGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        edge_dim=edge_dim,
        use_edge_mask=True
    )
    
    # Train model using 3-phase approach
    trained_model, embeddings, test_results = train_model_3_phase(
        model,
        data,
        num_epochs_phase1=100,
        num_epochs_phase2=100,
        num_epochs_phase3=30,
        lr=0.0001,
        test_ratio=0.15,
        val_ratio=0.15,
        results_dir=results_dir
    )
    
    # Extract key metrics
    result = {
        'test_accuracy': test_results['test_accuracy'],
        'test_roc_auc': test_results['test_roc_auc'],
        'test_precision': test_results['test_precision'],
        'test_recall': test_results['test_recall'],
        'test_f1': test_results['test_f1'],
        'test_mcc': test_results['test_mcc'],
        'test_specificity': test_results['test_specificity'],
        'test_sensitivity': test_results['test_sensitivity'],
        'best_threshold': test_results['best_threshold']
    }
    
    return result

def save_intermediate_results(results, experiment_dir):
    """Save intermediate results after each experiment"""
    with open(os.path.join(experiment_dir, 'intermediate_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

def aggregate_experiment_results(all_results, experiment_dir):
    """Aggregate results from multiple experiments"""
    
    # Filter successful runs
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    if not successful_results:
        return None
    
    # Metrics to aggregate
    metrics = ['test_accuracy', 'test_roc_auc', 'test_precision', 'test_recall', 
               'test_f1', 'test_mcc', 'test_specificity', 'test_sensitivity', 'best_threshold']
    
    # Calculate statistics
    aggregated = {
        'num_successful_runs': len(successful_results),
        'num_total_runs': len(all_results),
        'metrics': {}
    }
    
    for metric in metrics:
        values = [r[metric] for r in successful_results if metric in r]
        if values:
            aggregated['metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values
            }
    
    # Save aggregated results
    with open(os.path.join(experiment_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Create results dataframe
    df = pd.DataFrame(successful_results)
    df.to_csv(os.path.join(experiment_dir, 'all_results.csv'), index=False)
    
    return aggregated

def create_master_aggregation(results_base_dir):
    """Create master aggregation from ALL experiments in the base directory"""
    print(f"\n{'='*80}")
    print(f"CREATING MASTER AGGREGATION")
    print(f"{'='*80}")
    
    # Collect all results from all experiments
    all_results = []
    experiment_folders = []
    
    if os.path.exists(results_base_dir):
        for exp_folder in os.listdir(results_base_dir):
            # Skip the master_aggregation folder itself
            if exp_folder == "master_aggregation":
                continue
                
            exp_path = os.path.join(results_base_dir, exp_folder)
            if os.path.isdir(exp_path):
                # Check for seed directories
                for item in os.listdir(exp_path):
                    item_path = os.path.join(exp_path, item)
                    if item.startswith('seed_') and os.path.isdir(item_path):
                        try:
                            seed = int(item.split('_')[1])
                            test_results_path = os.path.join(item_path, 'test_results.json')
                            if os.path.exists(test_results_path):
                                with open(test_results_path, 'r') as f:
                                    result = json.load(f)
                                    result['seed'] = seed
                                    result['status'] = 'success'
                                    result['experiment_folder'] = exp_folder
                                    all_results.append(result)
                                    if exp_folder not in experiment_folders:
                                        experiment_folders.append(exp_folder)
                        except Exception as e:
                            print(f"  Error reading seed_{seed} in {exp_folder}: {e}")
    
    print(f"Found {len(all_results)} total completed experiments across {len(experiment_folders)} folders")
    
    if len(all_results) > 0:
        # Create master aggregation directory
        master_dir = os.path.join(results_base_dir, "master_aggregation")
        os.makedirs(master_dir, exist_ok=True)
        
        # Save all individual results
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv(os.path.join(master_dir, 'all_experiments.csv'), index=False)
        
        # Create aggregated statistics
        master_aggregated = aggregate_experiment_results(all_results, master_dir)
        
        # Create comprehensive visualizations
        create_multi_seed_visualizations(master_aggregated, master_dir)
        create_comprehensive_analysis(all_results, master_dir)
        
        print(f"\nMASTER AGGREGATION (All {len(all_results)} seeds)")
        print_aggregated_results(master_aggregated)
        
        # Save master summary
        summary = {
            'total_experiments': len(all_results),
            'experiment_folders': experiment_folders,
            'seed_range': [min(r['seed'] for r in all_results), max(r['seed'] for r in all_results)],
            'aggregated_metrics': master_aggregated,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open(os.path.join(master_dir, 'master_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nMaster aggregation saved to: {master_dir}/")
    else:
        print("No completed experiments found for aggregation!")

def create_comprehensive_analysis(all_results, master_dir):
    """Create comprehensive analysis plots for all experiments"""
    print("  - Creating comprehensive analysis plots")
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    
    # 1. Create a comprehensive metrics plot with individual experiment points
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    metrics = ['test_accuracy', 'test_roc_auc', 'test_precision', 
               'test_recall', 'test_f1', 'test_mcc']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Create box plot with individual points
        data = df[metric].values
        
        # Box plot
        bp = ax.boxplot([data], positions=[1], widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Add individual points with jitter
        y = data
        x = np.random.normal(1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.5, s=50, color='navy')
        
        # Add mean line
        mean_val = np.mean(data)
        ax.axhline(y=mean_val, color='green', linestyle='--', linewidth=2, alpha=0.7)
        
        # Statistics text
        stats_text = f"Mean: {mean_val:.4f}\nStd: {np.std(data):.4f}\nMin: {np.min(data):.4f}\nMax: {np.max(data):.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1])
        ax.set_xticklabels(['All Seeds'])
        ax.set_ylabel(metric.replace('test_', '').replace('_', ' ').title())
        ax.set_title(f"{metric.replace('test_', '').replace('_', ' ').title()} Distribution")
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Metrics Distribution Across {len(all_results)} Experiments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(master_dir, 'comprehensive_metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create experiment timeline plot
    if 'experiment_folder' in df.columns:
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Sort by seed to show progression
        df_sorted = df.sort_values('seed')
        
        # Plot F1 score over experiments
        ax.plot(df_sorted['seed'], df_sorted['test_f1'], 'o-', alpha=0.7, markersize=8)
        
        # Add rolling mean
        window_size = min(10, len(df_sorted) // 3)
        if window_size > 1:
            rolling_mean = df_sorted['test_f1'].rolling(window=window_size, center=True).mean()
            ax.plot(df_sorted['seed'], rolling_mean, 'r-', linewidth=3, 
                    label=f'Rolling Mean (window={window_size})')
        
        ax.set_xlabel('Seed', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('F1 Score Progression Across All Experiments', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(master_dir, 'f1_progression_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Create correlation heatmap between metrics
    metrics_df = df[metrics]
    correlation_matrix = metrics_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": .8},
                fmt='.3f', annot_kws={'size': 10})
    plt.title('Correlation Between Performance Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(master_dir, 'metrics_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create performance summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary statistics table
    summary_data = []
    for metric in metrics:
        values = df[metric].values
        summary_data.append([
            metric.replace('test_', '').replace('_', ' ').title(),
            f"{np.mean(values):.4f}",
            f"{np.std(values):.4f}",
            f"{np.median(values):.4f}",
            f"{np.min(values):.4f}",
            f"{np.max(values):.4f}",
            f"{np.percentile(values, 25):.4f}",
            f"{np.percentile(values, 75):.4f}"
        ])
    
    table = ax.table(cellText=summary_data,
                     colLabels=['Metric', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Q1', 'Q3'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style the header
    for i in range(8):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(8):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title(f'Performance Summary Statistics ({len(all_results)} experiments)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(master_dir, 'performance_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Comprehensive analysis plots saved to: {master_dir}/")

def create_multi_seed_visualizations(aggregated_results, experiment_dir):
    """Create visualizations for multi-seed experiment results"""
    
    viz_dir = os.path.join(experiment_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    metrics_data = aggregated_results['metrics']
    
    # 1. Box plots for all metrics in one figure on the same axis
    plt.figure(figsize=(15, 10))
    metrics_to_plot = ['test_accuracy', 'test_roc_auc', 'test_precision', 
                       'test_recall', 'test_f1', 'test_mcc', 'test_specificity']
    
    data_to_plot = []
    labels = []
    
    for metric in metrics_to_plot:
        if metric in metrics_data:
            data_to_plot.append(metrics_data[metric]['values'])
            labels.append(metric.replace('test_', '').replace('_', ' ').title())
    
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    
    plt.ylabel('Score', fontsize=12)
    plt.title('Distribution of Metrics Across All Seeds', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'metrics_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart with error bars
    plt.figure(figsize=(12, 8))
    
    metric_names = []
    means = []
    stds = []
    
    for metric in metrics_to_plot:
        if metric in metrics_data:
            metric_names.append(metric.replace('test_', '').replace('_', ' ').title())
            means.append(metrics_data[metric]['mean'])
            stds.append(metrics_data[metric]['std'])
    
    x = np.arange(len(metric_names))
    bars = plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(x, metric_names, rotation=45)
    plt.ylabel('Score', fontsize=12)
    plt.title('Mean Performance Metrics with Standard Deviation', fontsize=14, fontweight='bold')
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mean_metrics_barplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Metric correlation heatmap
    if len(aggregated_results['metrics']['test_accuracy']['values']) > 3:
        plt.figure(figsize=(10, 8))
        
        # Create dataframe for correlation
        data_dict = {}
        for metric in metrics_to_plot:
            if metric in metrics_data:
                data_dict[metric.replace('test_', '')] = metrics_data[metric]['values']
        
        df = pd.DataFrame(data_dict)
        correlation_matrix = df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Performance Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'metrics_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()

def print_aggregated_results(aggregated_results):
    """Print aggregated results in a formatted way"""
    
    print("\n" + "="*80)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*80)
    print(f"Successful runs: {aggregated_results['num_successful_runs']}/{aggregated_results['num_total_runs']}")
    print("\nMetrics (mean ± std):")
    print("-"*60)
    
    for metric, values in aggregated_results['metrics'].items():
        print(f"{metric:20s}: {values['mean']:.4f} ± {values['std']:.4f} "
              f"[{values['min']:.4f}, {values['max']:.4f}]")
    
    # Calculate 95% confidence intervals
    print("\n95% Confidence Intervals:")
    print("-"*60)
    
    from scipy import stats as scipy_stats
    
    for metric, values in aggregated_results['metrics'].items():
        mean = values['mean']
        std = values['std']
        n = len(values['values'])
        
        if n > 1:
            ci = scipy_stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
            print(f"{metric:20s}: [{ci[0]:.4f}, {ci[1]:.4f}]")


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set global seed
set_seed(SEED)

def find_optimal_threshold_balanced(y_true, y_scores, metric='balanced_accuracy'):
    """GPU-optimized threshold finding"""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    
    # Convert to GPU tensors for faster computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_scores_tensor = torch.tensor(y_scores, device=device)
    y_true_tensor = torch.tensor(y_true, device=device)
    
    thresholds = torch.linspace(y_scores_tensor.min(), y_scores_tensor.max(), 100, device=device)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_tensor = (y_scores_tensor > threshold).float()
        
        # GPU-accelerated metric calculation
        if metric == 'balanced_accuracy':
            tp = torch.sum((y_true_tensor == 1) & (y_pred_tensor == 1)).float()
            tn = torch.sum((y_true_tensor == 0) & (y_pred_tensor == 0)).float()
            fp = torch.sum((y_true_tensor == 0) & (y_pred_tensor == 1)).float()
            fn = torch.sum((y_true_tensor == 1) & (y_pred_tensor == 0)).float()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=device)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else torch.tensor(0.0, device=device)
            score = (sensitivity + specificity) / 2
        else:
            # Fallback to sklearn for other metrics
            y_pred_cpu = y_pred_tensor.cpu().numpy()
            y_true_cpu = y_true_tensor.cpu().numpy()
            if metric == 'f1':
                score = f1_score(y_true_cpu, y_pred_cpu, zero_division=0)
            else:
                score = balanced_accuracy_score(y_true_cpu, y_pred_cpu)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold.item()
    
    return best_threshold, best_score

def calculate_balanced_metrics(y_true, y_pred, y_scores):
    """GPU-accelerated metrics calculation where possible"""
    from sklearn.metrics import (
        balanced_accuracy_score, accuracy_score, precision_score, recall_score, 
        f1_score, matthews_corrcoef, average_precision_score, roc_auc_score, 
        confusion_matrix
    )
    
    # Use GPU for basic calculations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_true_tensor = torch.tensor(y_true, device=device)
    y_pred_tensor = torch.tensor(y_pred, device=device)
    
    # GPU-accelerated confusion matrix elements
    tp = torch.sum((y_true_tensor == 1) & (y_pred_tensor == 1)).item()
    tn = torch.sum((y_true_tensor == 0) & (y_pred_tensor == 0)).item()
    fp = torch.sum((y_true_tensor == 0) & (y_pred_tensor == 1)).item()
    fn = torch.sum((y_true_tensor == 1) & (y_pred_tensor == 0)).item()
    
    # GPU-accelerated basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Fallback to sklearn for complex metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc,
        'average_precision': avg_precision,
        'roc_auc': roc_auc,
        'sensitivity': recall,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'pos_precision': precision,
        'neg_precision': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'pos_recall': recall,
        'neg_recall': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    }
    
    return metrics

class EdgeGCNConv(MessagePassing):
    """
    MATHEMATICAL FORMULAS:
    
    1. MESSAGE FUNCTION:
       m_{ij}^{(l)} = σ(W_e * e_{ij}) * (D^{-1/2}_{ii} * D^{-1/2}_{jj}) * W^{(l)} * h_j^{(l)}
       
       Where:
       - m_{ij}^{(l)} = message from node j to node i at layer l
       - σ = sigmoid activation
       - W_e = edge feature transformation matrix
       - e_{ij} = edge attributes (pChEMBL values)
       - D^{-1/2}_{ii}, D^{-1/2}_{jj} = normalization factors (inverse sqrt of degrees)
       - W^{(l)} = node feature transformation matrix at layer l
       - h_j^{(l)} = node features of neighbor j at layer l
       - * = element-wise multiplication
    
    2. AGGREGATION FUNCTION:
       h_i^{(l+1)} = Σ_{j∈N(i)} m_{ij}^{(l)}
       
       Where:
       - h_i^{(l+1)} = updated features for node i at layer l+1
       - N(i) = neighbors of node i (including self-loop)
       - Σ = summation (add aggregation)
    
    3. NORMALIZATION:
       D^{-1/2}_{ii} * D^{-1/2}_{jj} where D_{ii} = |N(i)| (degree of node i)
    """
    
    def __init__(self, in_channels, out_channels, edge_dim=2):
        super(EdgeGCNConv, self).__init__(aggr='add')  
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_lin = nn.Linear(edge_dim, out_channels)
        self.use_self_loop_weights = False
        
    def forward(self, x, edge_index, edge_attr=None):
        # store original edge count and transformed features
        original_edge_count = edge_index.size(1)
        
        # transform node features
        x_transformed = self.lin(x)
        
        # add self-loops
        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # handle edge attributes
        if edge_attr is not None and edge_attr.size(0) > 0:
            num_self_loops = edge_index.size(1) - original_edge_count
            self_loop_attr = torch.ones((num_self_loops, edge_attr.size(1)), 
                                       dtype=edge_attr.dtype, device=edge_attr.device) * 1.0
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        else:
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=x_transformed.dtype, device=x_transformed.device) * 1.0
        
        # compute normalization
        row, col = edge_index
        deg = degree(col, x_transformed.size(0), dtype=x_transformed.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # add self-loops directly without edge weighting
        if not self.use_self_loop_weights:
            # separate self-loop edges from neighbor edges
            self_loop_mask = row == col
            neighbor_mask = ~self_loop_mask
            
            # process neighbor messages with edge weights
            if neighbor_mask.any():
                neighbor_messages = self.propagate(
                    edge_index[:, neighbor_mask], 
                    x=x_transformed, 
                    norm=norm[neighbor_mask], 
                    edge_attr=edge_attr[neighbor_mask]
                )
            else:
                neighbor_messages = torch.zeros_like(x_transformed)
            
            # add self-loop contributions directly (unweighted, but normalized)
            if self_loop_mask.any():
                self_loop_norm = norm[self_loop_mask]
                self_contributions = x_transformed * self_loop_norm.view(-1, 1)
            else:
                self_contributions = torch.zeros_like(x_transformed)
            
            return neighbor_messages + self_contributions
        else:
            return self.propagate(edge_index, x=x_transformed, norm=norm, edge_attr=edge_attr)
    
    def message(self, x_j, norm, edge_attr):
        """      
        THIS IS THE CORE GRAPH CONVOLUTION OPERATION!
        
        MATHEMATICAL FORMULA IMPLEMENTED HERE:
        m_{ij}^{(l)} = σ(W_e * e_{ij}) ⊙ (D^{-1/2}_{ii} * D^{-1/2}_{jj}) * h_j^{(l)}
        """
        edge_weight = torch.sigmoid(self.edge_lin(edge_attr))
        return norm.view(-1, 1) * x_j * edge_weight

class SoftEdgeMask(nn.Module):
    """
    Enhanced edge masking with learned attention scores from node embeddings.
    
    Key improvements:
    1. Learned attention mechanism using compound and protein embeddings
    2. Multi-head attention for richer representation
    3. Progressive masking strategy to prevent overfitting
    4. Temperature-based gating for stable training
    """
    
    def __init__(self, compound_dim, protein_dim, hidden_dim=128, num_heads=4, top_k_ratio=0.15):
        super(SoftEdgeMask, self).__init__()
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.top_k_ratio = top_k_ratio
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention components for learning edge importance
        self.compound_query = nn.Linear(compound_dim, hidden_dim)
        self.protein_key = nn.Linear(protein_dim, hidden_dim)
        self.protein_value = nn.Linear(protein_dim, hidden_dim)
        
        # Additional projection for protein queries (for bidirectional attention)
        self.protein_query = nn.Linear(protein_dim, hidden_dim)
        
        # Attention projection heads - now process concatenated query-key pairs
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            # Create layers for Xavier initialization
            layer1 = nn.Linear(self.head_dim * 2, self.head_dim)
            layer2 = nn.Linear(self.head_dim, self.head_dim // 2)
            layer3 = nn.Linear(self.head_dim // 2, 1)
            
            # Apply Xavier initialization with higher gain
            nn.init.xavier_uniform_(layer1.weight)
            nn.init.xavier_uniform_(layer2.weight)
            nn.init.xavier_uniform_(layer3.weight)
            
            attention_head = nn.Sequential(
                layer1,
                nn.LayerNorm(self.head_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                layer2,
                nn.GELU(),
                layer3
            )
            self.attention_heads.append(attention_head)
        
        # INITIALIZE interaction_projection HERE IN __init__
        self.interaction_projection = nn.ModuleList([
            nn.Linear(self.head_dim * 3, self.head_dim * 2) 
            for _ in range(self.num_heads)
        ])
                
        # INITIALIZE ALL LEARNABLE PARAMETERS IN __init__
        self.head_importance = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)  # Learnable temperature
                
        # Performance tracking for adaptive behavior
        self.register_buffer('training_step', torch.tensor(0))
        self.register_buffer('performance_history', torch.zeros(20))
        self.register_buffer('performance_idx', torch.tensor(0))
        
        # Efficient edge caching
        self.register_buffer('existing_edges_cache', None, persistent=False)
        self.cache_valid = False
        
        # Store attention data for visualization
        self.register_buffer('last_attention_scores', None, persistent=False)
        self.register_buffer('last_selected_pairs', None, persistent=False)
        self.register_buffer('last_edge_weights', None, persistent=False)
        
    def forward(self, compound_embeddings, protein_embeddings, existing_edges, node_types, num_compounds, num_proteins, epoch=0):
        """
        Forward pass of the edge masking module.
        
        CRITICAL: This should work on ANY graph structure, not just training graphs.
        The learned attention mechanism identifies important edges dynamically.
        
        Args:
            compound_embeddings: Tensor of compound node embeddings [num_compounds, hidden_dim]
            protein_embeddings: Tensor of protein node embeddings [num_proteins, hidden_dim]
            existing_edges: Existing edges (not used in current implementation)
            node_types: Node type indicators (not used in current implementation)
            num_compounds: Number of compound nodes
            num_proteins: Number of protein nodes
            epoch: Current training epoch for progressive masking (use high value for evaluation)
            
        Returns:
            edge_indices: Selected edge indices [2, num_edges]
            edge_weights: Weights for selected edges [num_edges]
        """
        # Compute attention scores between compounds and proteins
        # This is LEARNED and works on any compound/protein embeddings
        attention_scores = self.compute_attention_scores(compound_embeddings, protein_embeddings)
        
        # Store attention data for visualization
        self.last_attention_scores = attention_scores.detach().clone()
        
        # Select edges based on attention scores with progressive strategy
        edge_indices, edge_weights = self.select_edges_progressive(
            attention_scores, epoch, num_compounds, num_proteins
        )
        
        # Store selected pairs and weights for visualization
        if edge_indices.size(1) > 0:
            # Extract compound-protein pairs (first half of bidirectional edges)
            num_unidirectional = edge_indices.size(1) // 2
            selected_compounds = edge_indices[0, :num_unidirectional]
            selected_proteins = edge_indices[1, :num_unidirectional] - num_compounds
            
            self.last_selected_pairs = torch.stack([selected_compounds, selected_proteins], dim=0)
            self.last_edge_weights = edge_weights[:num_unidirectional].detach().clone()
        else:
            self.last_selected_pairs = torch.empty((2, 0), dtype=torch.long, device=attention_scores.device)
            self.last_edge_weights = torch.empty((0,), dtype=torch.float, device=attention_scores.device)
        
        return edge_indices, edge_weights

    def compute_attention_scores(self, compound_embeddings, protein_embeddings):
        """Compute learned attention scores with MORE VARIANCE"""
        batch_size_c = compound_embeddings.size(0)
        batch_size_p = protein_embeddings.size(0)
        device = compound_embeddings.device
        
        # Transform embeddings
        queries_c = self.compound_query(compound_embeddings)  # [C, hidden_dim]
        keys_p = self.protein_key(protein_embeddings)        # [P, hidden_dim]
        
        # Layer normalization
        queries_c = F.layer_norm(queries_c, [self.hidden_dim])
        keys_p = F.layer_norm(keys_p, [self.hidden_dim])
        
        # Reshape for multi-head attention
        queries_c = queries_c.view(batch_size_c, self.num_heads, self.head_dim)
        keys_p = keys_p.view(batch_size_p, self.num_heads, self.head_dim)
        
        # Initialize attention scores tensor
        all_attention_scores = torch.zeros(batch_size_c, batch_size_p, self.num_heads, device=device)
        
        # Compute attention for each head
        for head in range(self.num_heads):
            # Get query and key for this head
            q_c = queries_c[:, head, :]  # [C, head_dim]
            k_p = keys_p[:, head, :]      # [P, head_dim]
            
            # Expand for pairwise computation
            q_c_expanded = q_c.unsqueeze(1).expand(-1, batch_size_p, -1)  # [C, P, head_dim]
            k_p_expanded = k_p.unsqueeze(0).expand(batch_size_c, -1, -1)  # [C, P, head_dim]
            
            # Concatenate query and key for each pair
            concat_features = torch.cat([q_c_expanded, k_p_expanded], dim=-1)  # [C, P, head_dim*2]
            concat_flat = concat_features.view(-1, self.head_dim * 2)  # [C*P, head_dim*2]
            
            # Pass through attention head network
            attention_scores = self.attention_heads[head](concat_flat).view(batch_size_c, batch_size_p)  # [C, P]
            
            # Store scores for this head
            all_attention_scores[:, :, head] = attention_scores
        
        # Apply softmax to head importance weights
        head_weights = F.softmax(self.head_importance, dim=0)
        
        # Weighted combination of all heads
        final_scores = torch.zeros(batch_size_c, batch_size_p, device=device)
        for h in range(self.num_heads):
            final_scores += all_attention_scores[:, :, h] * head_weights[h]
        
        # Apply temperature scaling before sigmoid for sharper scores
        final_scores = final_scores / self.temperature
        
        # Apply sigmoid to normalize to [0,1] range
        final_scores = torch.sigmoid(final_scores)

        return final_scores

    def select_edges_progressive(self, attention_scores, epoch, num_compounds, num_proteins):
        """Progressive edge selection strategy - SIMPLIFIED LINEAR GROWTH"""
        device = attention_scores.device

        # Use raw attention scores directly
        scaled_scores = attention_scores

        # SUPER SIMPLE: Just linear growth
        edges_per_epoch = 1000  # Add 200 edges every epoch
        num_edges = edges_per_epoch * (epoch + 1)  # Epoch 0: 200, Epoch 1: 400, etc.
        
        # Simple top-k selection
        flat_scores = scaled_scores.flatten()
        top_k = min(num_edges, flat_scores.size(0))  # Only limit by total possible
        
        # Get top scoring edges
        top_scores, top_indices = torch.topk(flat_scores, top_k, sorted=False)
        
        # Convert flat indices back to 2D coordinates
        selected_compounds = top_indices // num_proteins
        selected_proteins = top_indices % num_proteins
        selected_scores = top_scores

        # Use raw attention scores as edge weights
        edge_weights = selected_scores
        
        # Create bidirectional edges
        compound_indices = selected_compounds
        protein_indices = selected_proteins + num_compounds

        edge_indices = torch.stack([
            torch.cat([compound_indices, protein_indices]),
            torch.cat([protein_indices, compound_indices])
        ], dim=0)

        edge_weights_bi = torch.cat([edge_weights, edge_weights])

        return edge_indices, edge_weights_bi

class HomogeneousSoftEdgeMask(nn.Module):
    """
    Soft edge masking for homogeneous interactions (protein-protein or compound-compound).
    Uses self-attention mechanism to identify important within-type interactions.
    """
    
    def __init__(self, node_dim, hidden_dim=128, num_heads=4, interaction_type='protein'):
        super(HomogeneousSoftEdgeMask, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.interaction_type = interaction_type  # 'protein' or 'compound'
        
        # Self-attention components
        self.query = nn.Linear(node_dim, hidden_dim)
        self.key = nn.Linear(node_dim, hidden_dim)
        
        # Attention heads for self-interactions - improved architecture
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            # Create layers for Xavier initialization
            layer1 = nn.Linear(self.head_dim * 2, self.head_dim)
            layer2 = nn.Linear(self.head_dim, 1)
            
            # Apply Xavier initialization with higher gain
            nn.init.xavier_uniform_(layer1.weight, gain=2.0)
            nn.init.xavier_uniform_(layer2.weight, gain=2.0)
            
            # Initialize biases
            #nn.init.constant_(layer1.bias, 0.1)
            #nn.init.constant_(layer2.bias, 0.1)
            
            attention_head = nn.Sequential(
                layer1,
                nn.LayerNorm(self.head_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                layer2
            )
            self.attention_heads.append(attention_head)
        
        # Progressive masking parameters
        self.head_importance = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        
    def compute_self_attention_scores(self, node_embeddings, batch_size=100):
        """Compute self-attention scores for homogeneous interactions with BATCHED computation for memory efficiency"""
        num_nodes = node_embeddings.size(0)
        device = node_embeddings.device
        
        # Skip computation if too few nodes
        if num_nodes < 2:
            return torch.zeros((num_nodes, num_nodes), device=device)
        
        # For large number of nodes, use batched computation
        if num_nodes > 500:  # Threshold for batched processing
            return self._compute_self_attention_scores_batched(node_embeddings, batch_size)
        
        # Original implementation for small number of nodes
        # Transform embeddings
        queries = self.query(node_embeddings)  # [N, hidden_dim]
        keys = self.key(node_embeddings)       # [N, hidden_dim]
        
        # Reshape for multi-head attention
        queries = queries.view(num_nodes, self.num_heads, self.head_dim)
        keys = keys.view(num_nodes, self.num_heads, self.head_dim)
        
        # Initialize attention scores
        all_attention_scores = torch.zeros(num_nodes, num_nodes, self.num_heads, device=device)
        
        # Compute attention for each head
        for head in range(self.num_heads):
            q_head = queries[:, head, :]  # [N, head_dim]
            k_head = keys[:, head, :]     # [N, head_dim]
            
            # Expand for pairwise computation
            q_expanded = q_head.unsqueeze(1).expand(-1, num_nodes, -1)  # [N, N, head_dim]
            k_expanded = k_head.unsqueeze(0).expand(num_nodes, -1, -1)  # [N, N, head_dim]
            
            # Concatenate for each pair
            pair_features = torch.cat([q_expanded, k_expanded], dim=-1)  # [N, N, head_dim*2]
            
            # Pass through attention head
            attention_scores = self.attention_heads[head](pair_features).squeeze(-1)  # [N, N]
            
            # Store scores
            all_attention_scores[:, :, head] = attention_scores

        # Apply softmax to head importance weights
        head_weights = F.softmax(self.head_importance, dim=0)
        
        # Combine attention scores from all heads
        final_scores = torch.zeros(num_nodes, num_nodes, device=device)
        for h in range(self.num_heads):
            final_scores += all_attention_scores[:, :, h] * head_weights[h]
        
        # Zero out diagonal (no self-loops)
        final_scores = final_scores * (1 - torch.eye(num_nodes, device=device))

        # Apply sigmoid to normalize to [0,1] range
        final_scores = torch.sigmoid(final_scores)
          
        return final_scores
    
    def _compute_self_attention_scores_batched(self, node_embeddings, batch_size=100):
        """Batched computation of self-attention scores to avoid OOM on large graphs"""
        num_nodes = node_embeddings.size(0)
        device = node_embeddings.device
        
        #print(f"    Using batched attention computation for {num_nodes} nodes (batch_size={batch_size})")
        
        # Transform embeddings once
        queries = self.query(node_embeddings)  # [N, hidden_dim] 
        keys = self.key(node_embeddings)       # [N, hidden_dim]
        
        # Reshape for multi-head attention
        queries = queries.view(num_nodes, self.num_heads, self.head_dim)
        keys = keys.view(num_nodes, self.num_heads, self.head_dim)
        
        # Initialize output tensor - this is the memory bottleneck
        # Instead of storing full matrix, we'll accumulate scores incrementally
        
        # Apply softmax to head importance weights
        head_weights = F.softmax(self.head_importance, dim=0)
        
        # Process each head separately to save memory
        final_scores = None
        
        for head in range(self.num_heads):
            q_head = queries[:, head, :]  # [N, head_dim]
            k_head = keys[:, head, :]     # [N, head_dim]
            
            # Initialize scores for this head
            head_scores = torch.zeros(num_nodes, num_nodes, device=device)
            
            # Process rows in batches
            for i in range(0, num_nodes, batch_size):
                i_end = min(i + batch_size, num_nodes)
                batch_q = q_head[i:i_end]  # [batch, head_dim]
                
                # Process columns in batches
                for j in range(0, num_nodes, batch_size):
                    j_end = min(j + batch_size, num_nodes)
                    batch_k = k_head[j:j_end]  # [batch, head_dim]
                    
                    # Compute pairwise features for this batch
                    batch_q_exp = batch_q.unsqueeze(1).expand(-1, j_end - j, -1)  # [batch_i, batch_j, head_dim]
                    batch_k_exp = batch_k.unsqueeze(0).expand(i_end - i, -1, -1)  # [batch_i, batch_j, head_dim]
                    
                    # Concatenate
                    pair_features = torch.cat([batch_q_exp, batch_k_exp], dim=-1)  # [batch_i, batch_j, head_dim*2]
                    
                    # Pass through attention head
                    batch_scores = self.attention_heads[head](pair_features).squeeze(-1)  # [batch_i, batch_j]
                    
                    # Store in head_scores
                    head_scores[i:i_end, j:j_end] = batch_scores
                    
                    # Clear cache periodically
                    if j % (batch_size * 10) == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Accumulate weighted scores from this head
            if final_scores is None:
                final_scores = head_scores * head_weights[head]
            else:
                # Add in-place to save memory
                final_scores.add_(head_scores, alpha=head_weights[head].item())
            
            # Delete head_scores to free memory immediately
            del head_scores
            
            # Clear cache after each head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Zero out diagonal more efficiently by processing in batches
        # Instead of creating a full identity matrix, zero out diagonal elements directly
        for i in range(num_nodes):
            final_scores[i, i] = 0
        
        # Apply sigmoid
        final_scores = torch.sigmoid(final_scores)
        
        return final_scores
    
    def select_edges_progressive(self, attention_scores, epoch, num_nodes, node_offset=0):
        """Progressive edge selection for HOMOGENEOUS interactions - SIMPLIFIED LINEAR GROWTH"""
        device = attention_scores.device
        
        # Skip if too few nodes
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)
        
        # SIMPLE LINEAR GROWTH - 100 edges per epoch
        edges_per_epoch = 100
        num_edges = edges_per_epoch * (epoch + 1) 
        
        # Calculate total possible edges (undirected graph, no self-loops)
        total_possible = num_nodes * (num_nodes - 1) // 2
        
        # NO CAP - just ensure we don't exceed total possible
        num_edges = min(num_edges, total_possible)
        
        # Get upper triangular indices (to avoid duplicate edges)
        triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
        triu_scores = attention_scores[triu_indices[0], triu_indices[1]]
        
        # Simple top-k selection
        top_k = min(num_edges, triu_scores.size(0))
        if top_k == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)
        
        top_scores, top_indices = torch.topk(triu_scores, top_k, sorted=False)
        
        # Get the actual node indices
        selected_i = triu_indices[0][top_indices]
        selected_j = triu_indices[1][top_indices]
        
        # Apply node offset (for proper global indexing)
        selected_i = selected_i + node_offset
        selected_j = selected_j + node_offset
        
        # Use raw attention scores as edge weights
        edge_weights = top_scores
        
        # Create bidirectional edges
        edge_indices = torch.stack([
            torch.cat([selected_i, selected_j]),
            torch.cat([selected_j, selected_i])
        ], dim=0)
        
        edge_weights_bi = torch.cat([edge_weights, edge_weights])
        
        return edge_indices, edge_weights_bi

class DrugTargetGNN(nn.Module):
    """
    MATHEMATICAL FORMULAS:
    
    1. MULTI-LAYER GCN:
       H^{(l+1)} = σ(D^{-1/2} * Ã * D^{-1/2} * H^{(l)} * W^{(l)} ⊙ EdgeWeights)
       
       Where:
       - H^{(l)} = node feature matrix at layer l
       - Ã = adjacency matrix with self-loops
       - D = degree matrix
       - W^{(l)} = learnable weight matrix at layer l
       - σ = activation function (ReLU)
       - EdgeWeights = sigmoid(W_e * E) where E are edge features
    
    2. PROJECTION HEAD:
       z = ReLU(W₂ * ReLU(W₁ * h^{(final)}))
       
       Where:
       - h^{(final)} = final node embeddings from GNN
       - W₁, W₂ = projection matrices
       - z = projected embeddings for contrastive learning
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=3, edge_dim=2, use_edge_mask=False):
        super(DrugTargetGNN, self).__init__()
        
        self.use_edge_mask = use_edge_mask
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout_rates = [0.4, 0.4, 0.3]  # Increased dropout rates
        
        # first layer
        self.convs.append(EdgeGCNConv(input_dim, hidden_dim, edge_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(EdgeGCNConv(hidden_dim, hidden_dim, edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # output layer
        self.convs.append(EdgeGCNConv(hidden_dim, output_dim, edge_dim))
        
        # separate projection heads for compounds and proteins to prevent collapse
        self.compound_projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased from 0.2
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.protein_projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased from 0.2
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # just add a scaling factor for proteins
        self.protein_scale = nn.Parameter(torch.ones(1) * 2.0)  # Learnable scaling
        
        # soft edge mask for phase 2
        if use_edge_mask:
            self.edge_mask = SoftEdgeMask(
                compound_dim=hidden_dim,  # Use hidden_dim instead of output_dim
                protein_dim=hidden_dim,   # Use hidden_dim instead of output_dim
                hidden_dim=128,
                num_heads=4,
                top_k_ratio=0.5  # Increased from 0.15 to allow more edges
            )
            
            # Add homogeneous edge masks for protein-protein and compound-compound
            self.protein_edge_mask = HomogeneousSoftEdgeMask(
                node_dim=hidden_dim,
                hidden_dim=128,
                num_heads=4,
                interaction_type='protein'
            )
            
            self.compound_edge_mask = HomogeneousSoftEdgeMask(
                node_dim=hidden_dim,
                hidden_dim=128,
                num_heads=4,
                interaction_type='compound'
            )
        
        # Classifier head for phase 3
        self.pair_classifier = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_attr=None, node_types=None, apply_edge_mask=False, data=None):
        """
        GRAPH CONVOLUTION LAYERS:
        Each conv layer performs: H^{(l+1)} = σ(GCN(H^{(l}), A, E))
        where GCN is implemented in EdgeGCNConv
        """
        
        # apply GNN layers with graph convolution at each step
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            # Use layer-specific dropout rates
            dropout_rate = self.dropout_rates[i] if i < len(self.dropout_rates) else 0.3
        
        # Store intermediate embeddings before final layer for edge masking
        intermediate_x = x
        
        # final layer graph convolution (no batch norm or activation)
        x = self.convs[-1](x, edge_index, edge_attr)
        
        # apply enhanced edge masking if enabled
        if apply_edge_mask and self.use_edge_mask and node_types is not None and data is not None:
            compound_mask = node_types == 0
            protein_mask = node_types == 1
            
            # Use intermediate embeddings for edge masking
            compound_embeddings = intermediate_x[compound_mask]
            protein_embeddings = intermediate_x[protein_mask]
            
            # get existing compound-protein edges
            existing_cp_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
            if 'cp_edge_index' in data:
                existing_cp_edges = data['cp_edge_index']
            
            # apply enhanced edge masking with epoch information
            epoch = data.get('epoch', 0)
            
            # 1. Compound-protein edge masking (always enabled)
            cp_masked_edges, cp_edge_weights = self.edge_mask(
                compound_embeddings, protein_embeddings, existing_cp_edges,
                node_types, data['num_compounds'], data['num_proteins'], epoch=epoch
            )
            
            # 2. Protein-protein edge masking - ENABLED FROM START OF PHASE 2
            pp_masked_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
            pp_edge_weights = torch.empty(0, device=x.device)
            if protein_embeddings.size(0) > 1:  # Only if we have multiple proteins
                pp_attention_scores = self.protein_edge_mask.compute_self_attention_scores(protein_embeddings)
                pp_masked_edges, pp_edge_weights = self.protein_edge_mask.select_edges_progressive(
                    pp_attention_scores,
                    epoch,
                    protein_embeddings.size(0),
                    node_offset=data['num_compounds']
                )
            
            # 3. Compound-compound edge masking - ENABLED FROM START OF PHASE 2
            cc_masked_edges = torch.empty((2, 0), dtype=torch.long, device=x.device)
            cc_edge_weights = torch.empty(0, device=x.device)
            if compound_embeddings.size(0) > 1:  # Only if we have multiple compounds
                cc_attention_scores = self.compound_edge_mask.compute_self_attention_scores(compound_embeddings)
                cc_masked_edges, cc_edge_weights = self.compound_edge_mask.select_edges_progressive(
                    cc_attention_scores,
                    epoch,
                    compound_embeddings.size(0),
                    node_offset=0
                )
            
            # Combine all masked edges
            all_masked_edges = [cp_masked_edges]
            all_edge_weights = [cp_edge_weights]
            
            if pp_masked_edges.size(1) > 0:
                all_masked_edges.append(pp_masked_edges)
                all_edge_weights.append(pp_edge_weights)
                
            if cc_masked_edges.size(1) > 0:
                all_masked_edges.append(cc_masked_edges)
                all_edge_weights.append(cc_edge_weights)
            
            if len(all_masked_edges) > 0 and all_masked_edges[0].size(1) > 0:
                combined_masked_edges = torch.cat(all_masked_edges, dim=1)
                combined_edge_weights = torch.cat(all_edge_weights, dim=0)
                
                # Create edge attributes for all masked edges
                masked_edge_attr = torch.ones((combined_masked_edges.size(1), edge_attr.size(1) if edge_attr is not None else 1), 
                                             dtype=x.dtype, device=x.device)
                
                # Apply edge weights with MORE CONSERVATIVE scaling for stability
                masked_edge_attr = masked_edge_attr * combined_edge_weights.unsqueeze(1)

                # IMPORTANT: Keep original edges and ADD new masked edges
                combined_edge_index = torch.cat([edge_index, combined_masked_edges], dim=1)
                if edge_attr is not None:
                    combined_edge_attr = torch.cat([edge_attr, masked_edge_attr], dim=0)
                else:
                    # Create edge attributes for original edges if they don't exist
                    original_edge_attr = torch.ones((edge_index.size(1), 1), dtype=x.dtype, device=x.device)
                    combined_edge_attr = torch.cat([original_edge_attr, masked_edge_attr], dim=0)
                
                # Re-run the final convolution with combined edges
                x = self.convs[-1](intermediate_x, combined_edge_index, combined_edge_attr)
                
                # Log edge statistics - with percentages
                if epoch % 10 == 0:
                    total_possible_cp = data['num_compounds'] * data['num_proteins']
                    total_possible_pp = data['num_proteins'] * (data['num_proteins'] - 1) // 2
                    total_possible_cc = data['num_compounds'] * (data['num_compounds'] - 1) // 2
                    
                    cp_pct = (cp_masked_edges.size(1) / 2) / total_possible_cp * 100 if total_possible_cp > 0 else 0
                    pp_pct = (pp_masked_edges.size(1) / 2) / total_possible_pp * 100 if total_possible_pp > 0 else 0
                    cc_pct = (cc_masked_edges.size(1) / 2) / total_possible_cc * 100 if total_possible_cc > 0 else 0
                    
                    print(f"    Edge mask stats: CP={cp_masked_edges.size(1)//2} ({cp_pct:.2f}%), "
                          f"PP={pp_masked_edges.size(1)//2} ({pp_pct:.2f}%), "
                          f"CC={cc_masked_edges.size(1)//2} ({cc_pct:.2f}%)")

        # projection for contrastive learning with separate heads
        if node_types is not None:
            compound_mask = node_types == 0
            protein_mask = node_types == 1
            
            z = torch.zeros_like(x)
            if compound_mask.any():
                z[compound_mask] = self.compound_projection(x[compound_mask])
            if protein_mask.any():
                # scale protein embeddings to increase variance
                protein_emb = self.protein_projection(x[protein_mask])
                z[protein_mask] = protein_emb * self.protein_scale
        else:
            # compound projection if no node types
            z = self.compound_projection(x)
        
        return x, z

    def classify_pairs(self, z_compounds, z_proteins, compound_indices, protein_indices):
        """
        Classify compound-protein pairs using the learned classifier head.
        
        Args:
            z_compounds: Compound embeddings [num_compounds, embedding_dim]
            z_proteins: Protein embeddings [num_proteins, embedding_dim]
            compound_indices: Indices of compounds to classify
            protein_indices: Indices of proteins to classify
            
        Returns:
            logits: Classification logits
            probabilities: Classification probabilities (sigmoid of logits)
        """
        # Get embeddings for the specified pairs
        compound_embs = z_compounds[compound_indices]
        protein_embs = z_proteins[protein_indices]
        
        # Concatenate embeddings
        pair_embs = torch.cat([compound_embs, protein_embs], dim=1)
        
        # Pass through classifier
        logits = self.pair_classifier(pair_embs)
        
        # Handle dimension issues
        if logits.dim() > 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        
        probabilities = torch.sigmoid(logits)
        
        return logits, probabilities

class TripletContrastiveLoss(nn.Module):
    """
    FULLY VECTORIZED: GPU-optimized triplet loss with ADVANCED HARD NEGATIVE MINING
    WITH BALANCED LOSS WEIGHTING for better class balance
    """
    
    def __init__(self, margin=1.0, max_negatives_per_positive=20, batch_size=512, temperature=0.1, 
                 pos_weight=1.0, neg_weight=1.0, hard_negative_mining=True):
        super(TripletContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_negatives_per_positive = max_negatives_per_positive
        self.batch_size = batch_size
        self.temperature = temperature
        self.pos_weight = pos_weight  # Weight for positive pairs
        self.neg_weight = neg_weight  # Weight for negative pairs
        self.hard_negative_mining = hard_negative_mining  # enable/disable hard negative mining
    
    def set_class_weights(self, pos_weight, neg_weight):
        """Dynamically adjust class weights for balanced training"""
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def create_pair_embedding(self, compound_emb, protein_emb, method='concat'):
        """Create pair embedding from individual compound and protein embeddings"""
        if method == 'concat':
            return torch.cat([compound_emb, protein_emb], dim=-1)
        elif method == 'hadamard':
            min_dim = min(compound_emb.size(-1), protein_emb.size(-1))
            return compound_emb[..., :min_dim] * protein_emb[..., :min_dim]
        else:
            return torch.cat([compound_emb, protein_emb], dim=-1)
    
    def create_pair_embedding_batch(self, compound_embs, protein_embs, method='concat'):
        """VECTORIZED: Create multiple pair embeddings at once"""
        if method == 'concat':
            return torch.cat([compound_embs, protein_embs], dim=-1)
        elif method == 'hadamard':
            min_dim = min(compound_embs.size(-1), protein_embs.size(-1))
            return compound_embs[..., :min_dim] * protein_embs[..., :min_dim]
        else:
            return torch.cat([compound_embs, protein_embs], dim=-1)
    
    def forward(self, z_compounds, z_proteins, positive_pairs, apply_regularization=True):
        """FULLY VECTORIZED with ADVANCED HARD NEGATIVE MINING: Process large batches simultaneously on GPU"""
        device = z_compounds.device
        
        # normalize embeddings once
        z_compounds = F.normalize(z_compounds, p=2, dim=1)
        z_proteins = F.normalize(z_proteins, p=2, dim=1)
        
        # L2 regularization removed - not needed
        embedding_reg = 0.0
        
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Original: Pair embedding loss with advanced hard negative mining
        return self._pair_embedding_loss_advanced(z_compounds, z_proteins, positive_pairs, device)

    def _pair_embedding_loss_advanced(self, z_compounds, z_proteins, positive_pairs, device):
        """Advanced hard negative mining for pair embeddings - CORRECTED VERSION"""
        positive_set = set(positive_pairs)
        # Optional bool [C, P] mask (set by the trainers): pairs that may never be used as negatives.
        reserved = getattr(self, "reserved_mask", None)

        def blocked(c, p):
            return (c, p) in positive_set or (reserved is not None and bool(reserved[c, p]))

        max_batch_pairs = min(self.batch_size, len(positive_pairs))
        max_positives_per_anchor = 20
        batch_pairs = random.sample(positive_pairs, max_batch_pairs)
        
        # Create anchor pair embeddings
        anchor_c_indices = torch.tensor([c for c, p in batch_pairs], device=device)
        anchor_p_indices = torch.tensor([p for c, p in batch_pairs], device=device)
        anchor_compound_embs = z_compounds[anchor_c_indices]
        anchor_protein_embs = z_proteins[anchor_p_indices]
        anchor_pair_embs = self.create_pair_embedding_batch(anchor_compound_embs, anchor_protein_embs)
        anchor_pair_embs = F.normalize(anchor_pair_embs, p=2, dim=1)
        
        num_compounds = z_compounds.size(0)
        num_proteins = z_proteins.size(0)
        neg_pair_embs_list = []
        
        for i, (anchor_c, anchor_p) in enumerate(batch_pairs):
            anchor_pair = anchor_pair_embs[i]
            neg_candidates = []
            
            # 1. Generate candidates with same compound, different protein
            neg_protein_indices = [k for k in range(num_proteins) if not blocked(anchor_c, k) and k != anchor_p]
            if neg_protein_indices:
                # Generate ALL candidates and compute their similarities
                for neg_p in neg_protein_indices[:min(50, len(neg_protein_indices))]:  # Limit for efficiency
                    neg_pair_emb = self.create_pair_embedding(z_compounds[anchor_c], z_proteins[neg_p])
                    neg_pair_emb = F.normalize(neg_pair_emb, p=2, dim=0)
                    # Compute similarity to anchor PAIR
                    similarity = F.cosine_similarity(anchor_pair.unsqueeze(0), neg_pair_emb.unsqueeze(0)).item()
                    neg_candidates.append((neg_pair_emb, similarity, 'same_compound'))
            
            # 2. Generate candidates with same protein, different compound
            neg_compound_indices = [l for l in range(num_compounds) if not blocked(l, anchor_p) and l != anchor_c]
            if neg_compound_indices:
                for neg_c in neg_compound_indices[:min(50, len(neg_compound_indices))]:
                    neg_pair_emb = self.create_pair_embedding(z_compounds[neg_c], z_proteins[anchor_p])
                    neg_pair_emb = F.normalize(neg_pair_emb, p=2, dim=0)
                    similarity = F.cosine_similarity(anchor_pair.unsqueeze(0), neg_pair_emb.unsqueeze(0)).item()
                    neg_candidates.append((neg_pair_emb, similarity, 'same_protein'))
            
            # 3. Generate random candidates for diversity
            for _ in range(30):  # Generate some random candidates
                neg_c = random.randint(0, num_compounds - 1)
                neg_p = random.randint(0, num_proteins - 1)
                if not blocked(neg_c, neg_p):
                    neg_pair_emb = self.create_pair_embedding(z_compounds[neg_c], z_proteins[neg_p])
                    neg_pair_emb = F.normalize(neg_pair_emb, p=2, dim=0)
                    similarity = F.cosine_similarity(anchor_pair.unsqueeze(0), neg_pair_emb.unsqueeze(0)).item()
                    neg_candidates.append((neg_pair_emb, similarity, 'random'))
            
            # NOW SELECT HARD NEGATIVES BASED ON SIMILARITY
            if neg_candidates:
                # Sort by similarity (descending) - highest similarity = hardest negative
                neg_candidates.sort(key=lambda x: x[1], reverse=True)
                
                selected_negatives = []
                
                if self.hard_negative_mining:
                    # Select hardest negatives (70% hard, 30% random for diversity)
                    num_hard = min(int(self.max_negatives_per_positive * 0.7), len(neg_candidates))
                    num_random = self.max_negatives_per_positive - num_hard
                    
                    # Take the hardest negatives (highest similarity to anchor pair)
                    for j in range(num_hard):
                        if j < len(neg_candidates):
                            selected_negatives.append(neg_candidates[j][0])  # Just the embedding
                    
                    # Add some random negatives from the remaining candidates
                    remaining = neg_candidates[num_hard:]
                    if remaining and num_random > 0:
                        random_sample = random.sample(remaining, min(num_random, len(remaining)))
                        for neg_emb, _, _ in random_sample:
                            selected_negatives.append(neg_emb)
                else:
                    # If hard negative mining is disabled, just take random samples
                    random_sample = random.sample(neg_candidates, min(self.max_negatives_per_positive, len(neg_candidates)))
                    selected_negatives = [emb for emb, _, _ in random_sample]
                
                # Stack selected negatives
                if selected_negatives:
                    neg_embs = torch.stack(selected_negatives[:self.max_negatives_per_positive])
                else:
                    neg_embs = torch.zeros(1, anchor_pair_embs.size(1), device=device)
            else:
                neg_embs = torch.zeros(1, anchor_pair_embs.size(1), device=device)
            
            neg_pair_embs_list.append(neg_embs)
        
        # Pad negative embeddings to same size
        max_negs = max([embs.size(0) for embs in neg_pair_embs_list])
        neg_pair_embs = torch.zeros(len(batch_pairs), max_negs, anchor_pair_embs.size(1), device=device)
        for i, embs in enumerate(neg_pair_embs_list):
            neg_pair_embs[i, :embs.size(0)] = embs
        
        # ENHANCED POSITIVE SAMPLING with more diversity
        pos_indices = []
        for i, (anchor_c, anchor_p) in enumerate(batch_pairs):
            other_positives = [(c, p) for c, p in positive_pairs if (c, p) != (anchor_c, anchor_p)]
            if len(other_positives) > 0:
                sampled_positives = random.sample(other_positives, min(max_positives_per_anchor, len(other_positives)))
                pos_indices.extend(sampled_positives)
        
        if len(pos_indices) > 0:
            pos_c_indices = torch.tensor([c for c, p in pos_indices], device=device)
            pos_p_indices = torch.tensor([p for c, p in pos_indices], device=device)
            
            pos_compound_embs = z_compounds[pos_c_indices]
            pos_protein_embs = z_proteins[pos_p_indices]
            pos_pair_embs = self.create_pair_embedding_batch(pos_compound_embs, pos_protein_embs)
            pos_pair_embs = F.normalize(pos_pair_embs, p=2, dim=1)
        else:
            pos_pair_embs = anchor_pair_embs
        
        # ENHANCED SIMILARITY COMPUTATION with temperature scaling
        # Anchor-Positive similarities: [batch_size, num_positives]
        anchor_pos_sims = torch.mm(anchor_pair_embs, pos_pair_embs.t()) / self.temperature
        
        # Anchor-Negative similarities: [batch_size, max_negatives_per_anchor]
        anchor_neg_sims = torch.bmm(
            anchor_pair_embs.unsqueeze(1),  # [batch_size, 1, pair_dim]
            neg_pair_embs.transpose(1, 2)   # [batch_size, pair_dim, max_negatives]
        ).squeeze(1) / self.temperature    # [batch_size, max_negatives]
        
        # CONSERVATIVE TRIPLET LOSS ONLY (removed aggressive component)
        mean_neg_sims = anchor_neg_sims.mean(dim=1)   # [batch_size] - average negative
        mean_pos_sims = anchor_pos_sims.mean(dim=1)   # [batch_size] - average positive
        
        # Only conservative loss for stability
        conservative_loss = torch.clamp(
            self.margin + mean_neg_sims * self.neg_weight - mean_pos_sims * self.pos_weight,
            min=1e-4
        )
        
        # Average loss across batch
        final_loss = conservative_loss.mean()
        
        return final_loss

def create_negative_samples(positive_pairs, num_compounds, num_proteins, ratio=1):
    """Create negative samples by random pairing"""
    positive_set = set(positive_pairs)
    negative_pairs = []
    
    rng = np.random.RandomState(SEED)

    while len(negative_pairs) < len(positive_pairs) * ratio:
        c_idx = rng.randint(0, num_compounds - 1)
        p_idx = rng.randint(0, num_proteins - 1)
        
        if (c_idx, p_idx) not in positive_set:
            negative_pairs.append((c_idx, p_idx))

    print(f"Generated {len(negative_pairs)} negative pairs.")
    print(f"Sample negative pairs: {negative_pairs[:5]}")

    return negative_pairs

def split_interaction_data(data, test_ratio=0.15, val_ratio=0.15, random_state=42):
    """GPU-optimized data splitting with minimal CPU operations"""
    # Relabeled graphs (build_run_graphs.py) carry the original split restricted to their positives.
    if 'split_override' in data:
        s = data['split_override']
        return [tuple(p) for p in s['train']], [tuple(p) for p in s['val']], [tuple(p) for p in s['test']]
    # extract all positive pairs from COMPOUND-PROTEIN interactions only
    positive_pairs = []
    
    # use the separate compound-protein edge data to ensure we only get drug-target interactions
    if 'cp_edge_index' in data and data['cp_edge_index'].size > 0:
        cp_edge_list = data['cp_edge_index'].T
        for i in range(0, len(cp_edge_list), 2):  # skip duplicate edges (bidirectional)
            c_idx, p_idx = cp_edge_list[i]
            # convert protein index back to relative index within proteins
            positive_pairs.append((c_idx, p_idx - data['num_compounds']))
    else:
        # extract from combined edge list but filter by node types
        edge_list = data['edge_index'].T
        node_types = data['node_types']
        edge_types = data.get('edge_types', np.zeros(len(edge_list)))
        
        for i in range(0, len(edge_list), 2):  # skip duplicate edges
            if edge_types[i] == 0:  # only compound-protein edges
                c_idx, p_idx = edge_list[i]
                if node_types[c_idx] == 0 and node_types[p_idx] == 1:  # compound to protein
                    positive_pairs.append((c_idx, p_idx - data['num_compounds']))
    
    # split data
    train_val_pairs, test_pairs = train_test_split(positive_pairs, test_size=test_ratio, random_state=random_state)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=val_ratio/(1-test_ratio), random_state=random_state)

    # save the data split immediately after creating it
    split_data = {
        'train_pairs': train_pairs,
        'val_pairs': val_pairs,
        'test_pairs': test_pairs,
        'test_ratio': test_ratio,
        'val_ratio': val_ratio,
        'random_state': SEED
    }
    
    # save to pickle file
    os.makedirs("results", exist_ok=True)
    split_path = os.path.join("results", 'data_split.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"Data split saved to: {split_path}")
    
    print(f"\nContrastive Learning Data Split (COMPOUND-PROTEIN PAIRS ONLY):")
    print(f"  Total drug-target interactions: {len(positive_pairs)}")
    print(f"  Training pairs: {len(train_pairs)} ({len(train_pairs)/len(positive_pairs)*100:.1f}%)")
    print(f"  Validation pairs: {len(val_pairs)} ({len(val_pairs)/len(positive_pairs)*100:.1f}%)")
    print(f"  Test pairs: {len(test_pairs)} ({len(test_pairs)/len(positive_pairs)*100:.1f}%)")
    print(f"  Note: Compound-compound and protein-protein interactions are used for GNN message passing but NOT for contrastive learning")
    
    return train_pairs, val_pairs, test_pairs

def evaluate_triplet_model_standard(model, x, edge_index, edge_attr, node_types, test_pairs, data):
    """GPU-optimized evaluation with batched processing"""
    model.eval()
    device = x.device
    
    with torch.no_grad():
        # GPU-optimized forward pass - FIX CUDA WARNING
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                embeddings, z = model(x, edge_index, edge_attr)
        else:
            embeddings, z = model(x, edge_index, edge_attr)
        
        compound_mask = node_types == 0
        protein_mask = node_types == 1
        
        z_compounds = z[compound_mask]
        z_proteins = z[protein_mask]
        
        # GPU-optimized batch evaluation
        y_true = []
        y_scores = []
        
        # Larger batch size for evaluation (no gradients)
        eval_batch_size = 2000 if torch.cuda.is_available() else 500
        
        # Process positive pairs in batches
        for i in range(0, len(test_pairs), eval_batch_size):
            batch_end = min(i + eval_batch_size, len(test_pairs))
            batch_pairs = test_pairs[i:batch_end]
            
            if len(batch_pairs) > 0:
                # GPU tensor conversion
                c_indices = torch.tensor([c for c, p in batch_pairs], device=device, dtype=torch.long)
                p_indices = torch.tensor([p for c, p in batch_pairs], device=device, dtype=torch.long)
                
                # GPU batch embedding extraction
                anchor_compounds = z_compounds[c_indices]
                anchor_proteins = z_proteins[p_indices]
                
                # GPU batch similarity computation
                similarities = F.cosine_similarity(anchor_compounds, anchor_proteins, dim=1)
                
                y_true.extend([1] * len(batch_pairs))
                y_scores.extend(similarities.cpu().numpy())
        
        # GPU-optimized negative pair generation and evaluation
        num_compounds = data['num_compounds']
        num_proteins = data['num_proteins']
        positive_set = set(test_pairs)
        
        # Generate negative pairs more efficiently
        num_negatives = len(test_pairs)
        negative_pairs = []
        
        # GPU-accelerated negative sampling
        max_attempts = num_negatives * 5
        attempts = 0
        
        while len(negative_pairs) < num_negatives and attempts < max_attempts:
            # Generate batch of random pairs on GPU
            batch_size = min(1000, num_negatives - len(negative_pairs))
            
            c_candidates = torch.randint(0, num_compounds, (batch_size,), device=device)
            p_candidates = torch.randint(0, num_proteins, (batch_size,), device=device)
            
            # Check against positive set (CPU operation for set lookup)
            for c_idx, p_idx in zip(c_candidates.cpu().tolist(), p_candidates.cpu().tolist()):
                if (c_idx, p_idx) not in positive_set:
                    negative_pairs.append((c_idx, p_idx))
                    if len(negative_pairs) >= num_negatives:
                        break
            attempts += batch_size
        
        # Evaluate negative pairs in batches
        for i in range(0, len(negative_pairs), eval_batch_size):
            batch_end = min(i + eval_batch_size, len(negative_pairs))
            batch_pairs = negative_pairs[i:batch_end]
            
            if len(batch_pairs) > 0:
                c_indices = torch.tensor([c for c, p in batch_pairs], device=device, dtype=torch.long)
                p_indices = torch.tensor([p for c, p in batch_pairs], device=device, dtype=torch.long)
                
                batch_compounds = z_compounds[c_indices]
                batch_proteins = z_proteins[p_indices]
                similarities = F.cosine_similarity(batch_compounds, batch_proteins, dim=1)
                
                y_true.extend([0] * len(batch_pairs))
                y_scores.extend(similarities.cpu().numpy())
        
        return y_true, y_scores

def visualize_training_progress(train_losses, val_losses, val_aucs, save_path="training_progress.png"):
    """Visualize training progress with loss and AUC curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC curve
    ax2.plot(epochs, val_aucs, 'g-', label='Validation AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training progress plot saved to: {save_path}")

def visualize_score_distributions(y_true, y_scores, epoch=None, save_path="score_distributions.png"):
    """Visualize the distribution of scores for positive and negative pairs"""
    pos_scores = [s for s, t in zip(y_scores, y_true) if t == 1]
    neg_scores = [s for s, t in zip(y_scores, y_true) if t == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(pos_scores, bins=50, alpha=0.6, label=f'Positive pairs (n={len(pos_scores)})', color='green', density=True)
    plt.hist(neg_scores, bins=50, alpha=0.6, label=f'Negative pairs (n={len(neg_scores)})', color='red', density=True)
    
    # add vertical lines for means
    plt.axvline(np.mean(pos_scores), color='darkgreen', linestyle='--', linewidth=2, label=f'Pos mean: {np.mean(pos_scores):.3f}')
    plt.axvline(np.mean(neg_scores), color='darkred', linestyle='--', linewidth=2, label=f'Neg mean: {np.mean(neg_scores):.3f}')
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # add separation annotation
    separation = np.mean(pos_scores) - np.mean(neg_scores)
    plt.text(0.02, 0.98, f'Separation: {separation:.3f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Score distribution plot saved to: {save_path}")

def visualize_roc_pr_curves(y_true, y_scores, save_path="roc_pr_curves.png"):
    """Visualize ROC and PR curves"""
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC and PR curves saved to: {save_path}")

def visualize_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Visualize confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # add metrics annotations
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    plt.text(1.5, -0.1, metrics_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def visualize_embeddings_tsne(embeddings, node_types, save_path="embeddings_tsne.png", perplexity=30, n_iter=1000):
    """Visualize embeddings using t-SNE"""
    print("Computing t-SNE visualization")
    
    # sample if too many points
    max_points = 5000
    if len(embeddings) > max_points:
        indices = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings_sample = embeddings[indices]
        node_types_sample = node_types[indices]
    else:
        embeddings_sample = embeddings
        node_types_sample = node_types
    
    # ccompute t-SNE - FIX: use 'max_iter' instead of 'n_iter'
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_sample)
    
    # plot
    plt.figure(figsize=(10, 8))
    
    # separate compounds and proteins
    compound_mask = node_types_sample == 0
    protein_mask = node_types_sample == 1
    
    plt.scatter(embeddings_2d[compound_mask, 0], embeddings_2d[compound_mask, 1], 
                c='blue', alpha=0.6, s=20, label='Compounds')
    plt.scatter(embeddings_2d[protein_mask, 0], embeddings_2d[protein_mask, 1], 
                c='red', alpha=0.6, s=20, label='Proteins')
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to: {save_path}")

def visualize_embeddings_tsne_comprehensive(embeddings, node_types, y_true, y_scores, y_pred, test_pairs, z_compounds, z_proteins, results_dir="results", best_threshold=0.5):
    """Create comprehensive t-SNE visualizations of embeddings - SEPARATED INTO INDIVIDUAL PLOTS"""
    print("Computing comprehensive t-SNE visualizations (separate plots)")
    
    # Set random state for t-SNE reproducibility
    np.random.seed(SEED)
    
    compound_mask = node_types == 0
    protein_mask = node_types == 1
    
    # 1. t-SNE of all node embeddings
    print("  - Computing t-SNE for all node embeddings")
    plt.figure(figsize=(10, 8))
    tsne_all = TSNE(n_components=2, random_state=SEED, perplexity=30)
    embeddings_2d = tsne_all.fit_transform(embeddings)
    
    plt.scatter(embeddings_2d[compound_mask, 0], embeddings_2d[compound_mask, 1], 
               c='blue', alpha=0.6, s=30, label=f'Compounds (n={compound_mask.sum()})')
    plt.scatter(embeddings_2d[protein_mask, 0], embeddings_2d[protein_mask, 1], 
               c='red', alpha=0.6, s=30, label=f'Proteins (n={protein_mask.sum()})')
    plt.legend(fontsize=12)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tsne_all_nodes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE of compounds only
    print("  - Computing t-SNE for compounds only")
    compound_embeddings = embeddings[compound_mask]
    if len(compound_embeddings) > 0:
        plt.figure(figsize=(10, 8))
        tsne_compounds = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(compound_embeddings)-1))
        compounds_2d = tsne_compounds.fit_transform(compound_embeddings)
        
        plt.scatter(compounds_2d[:, 0], compounds_2d[:, 1], c='blue', alpha=0.6, s=50)
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'tsne_compounds.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. t-SNE of proteins only
    print("  - Computing t-SNE for proteins only")
    protein_embeddings = embeddings[protein_mask]
    if len(protein_embeddings) > 0:
        plt.figure(figsize=(10, 8))
        tsne_proteins = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(protein_embeddings)-1))
        proteins_2d = tsne_proteins.fit_transform(protein_embeddings)
        
        plt.scatter(proteins_2d[:, 0], proteins_2d[:, 1], c='red', alpha=0.6, s=50)
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'tsne_proteins.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. SIMPLIFIED t-SNE of drug-target pairs - SHOWING PREDICTIONS
    print("  - Computing t-SNE for drug-target pairs (showing predictions)")

    # Create pair embeddings with fixed random seed
    np.random.seed(SEED)
    pair_embeddings = []
    pair_predictions = []  # Changed from pair_labels to pair_predictions
    
    # Convert z_compounds and z_proteins to CPU for processing
    z_compounds_cpu = z_compounds.cpu().detach().numpy()
    z_proteins_cpu = z_proteins.cpu().detach().numpy()
    
    # Normalize embeddings
    z_compounds_norm = z_compounds_cpu / (np.linalg.norm(z_compounds_cpu, axis=1, keepdims=True) + 1e-8)
    z_proteins_norm = z_proteins_cpu / (np.linalg.norm(z_proteins_cpu, axis=1, keepdims=True) + 1e-8)
    
    # Add positive test pairs with their predictions
    for i, (c_idx, p_idx) in enumerate(test_pairs):
        if c_idx < len(z_compounds_cpu) and p_idx < len(z_proteins_cpu):
            pair_emb = np.concatenate([z_compounds_norm[c_idx], z_proteins_norm[p_idx]])
            pair_embeddings.append(pair_emb)
            # Use prediction instead of true label
            pair_predictions.append(y_pred[i])
    
    # Add negative pairs (same number as positive)
    num_compounds = len(z_compounds_cpu)
    num_proteins = len(z_proteins_cpu)
    positive_set = set(test_pairs)
    negative_pairs = []
    
    # Use seeded random for reproducible negative sampling
    rng = np.random.RandomState(SEED)
    attempts = 0
    neg_start_idx = len(test_pairs)  # Start index for negative pairs in y_pred
    
    rng = np.random.RandomState(SEED)

    while len(negative_pairs) < len(test_pairs) and attempts < len(test_pairs) * 10:
        c_idx = rng.randint(0, num_compounds)
        p_idx = rng.randint(0, num_proteins)
        if (c_idx, p_idx) not in positive_set:
            negative_pairs.append((c_idx, p_idx))
            pair_emb = np.concatenate([z_compounds_norm[c_idx], z_proteins_norm[p_idx]])
            pair_embeddings.append(pair_emb)
            # Use prediction for this negative pair
            if neg_start_idx + len(negative_pairs) - 1 < len(y_pred):
                pair_predictions.append(y_pred[neg_start_idx + len(negative_pairs) - 1])
            else:
                # Fallback if we don't have enough predictions
                pair_predictions.append(0)
        attempts += 1 
        
    print(f"    Created {len(pair_embeddings)} pair embeddings")
    
    if len(pair_embeddings) > 0:
        pair_embeddings = np.array(pair_embeddings)
        pair_predictions = np.array(pair_predictions)
        
        # SIMPLIFIED Plot: Show predictions instead of true labels
        plt.figure(figsize=(10, 8))
        
        perplexity = min(30, len(pair_embeddings)-1)
        tsne_pairs = TSNE(n_components=2, random_state=SEED, perplexity=perplexity)
        pairs_2d = tsne_pairs.fit_transform(pair_embeddings)
        
        pred_pos_mask = pair_predictions == 1
        pred_neg_mask = pair_predictions == 0
        
        # Scatter plot with predictions
        plt.scatter(pairs_2d[pred_pos_mask, 0], pairs_2d[pred_pos_mask, 1], 
                   c='darkgreen', alpha=0.7, s=60, 
                   label=f'Predicted Positive (n={pred_pos_mask.sum()})')
        plt.scatter(pairs_2d[pred_neg_mask, 0], pairs_2d[pred_neg_mask, 1], 
                   c='darkred', alpha=0.7, s=60,
                   label=f'Predicted Negative (n={pred_neg_mask.sum()})')
        
        # Add markers to show misclassifications
        true_labels = np.array([1] * len(test_pairs) + [0] * len(negative_pairs))[:len(pair_predictions)]
        misclassified = pair_predictions != true_labels
        if misclassified.any():
            plt.scatter(pairs_2d[misclassified, 0], pairs_2d[misclassified, 1], 
                       marker='x', s=100, c='black', alpha=0.8, linewidths=2,
                       label=f'Misclassified (n={misclassified.sum()})')
        
        plt.legend(fontsize=12, loc='best')
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'tsne_pairs_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. t-SNE of drug-target pairs - SHOWING TRUE LABELS
    print("  - Computing t-SNE for drug-target pairs (showing true labels)")
    print(f"    Creating pair embeddings with true labels...")

    # Create pair embeddings with true labels
    np.random.seed(SEED)
    true_pair_embeddings = []
    true_pair_labels = []
    
    # Add positive test pairs with true labels (always 1)
    for i, (c_idx, p_idx) in enumerate(test_pairs):
        if c_idx < len(z_compounds_cpu) and p_idx < len(z_proteins_cpu):
            pair_emb = np.concatenate([z_compounds_norm[c_idx], z_proteins_norm[p_idx]])
            true_pair_embeddings.append(pair_emb)
            true_pair_labels.append(1)  # True positive pairs
    
    # Add the same negative pairs but with true labels (always 0)
    for c_idx, p_idx in negative_pairs:
        if c_idx < len(z_compounds_cpu) and p_idx < len(z_proteins_cpu):
            pair_emb = np.concatenate([z_compounds_norm[c_idx], z_proteins_norm[p_idx]])
            true_pair_embeddings.append(pair_emb)
            true_pair_labels.append(0)  # True negative pairs
    
    print(f"    Created {len(true_pair_embeddings)} pair embeddings with true labels")
    
    if len(true_pair_embeddings) > 0:
        true_pair_embeddings = np.array(true_pair_embeddings)
        true_pair_labels = np.array(true_pair_labels)
        
        # Plot with true labels
        plt.figure(figsize=(10, 8))
        
        perplexity = min(30, len(true_pair_embeddings)-1)
        tsne_true_pairs = TSNE(n_components=2, random_state=SEED, perplexity=perplexity)
        true_pairs_2d = tsne_true_pairs.fit_transform(true_pair_embeddings)
        
        true_pos_mask = true_pair_labels == 1
        true_neg_mask = true_pair_labels == 0
        
        # Scatter plot with true labels
        plt.scatter(true_pairs_2d[true_pos_mask, 0], true_pairs_2d[true_pos_mask, 1], 
                   c='green', alpha=0.7, s=60, 
                   label=f'True Positive (n={true_pos_mask.sum()})')
        plt.scatter(true_pairs_2d[true_neg_mask, 0], true_pairs_2d[true_neg_mask, 1], 
                   c='red', alpha=0.7, s=60,
                   label=f'True Negative (n={true_neg_mask.sum()})')
        
        plt.legend(fontsize=12, loc='best')
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'tsne_pairs_true_labels.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("  t-SNE visualizations saved (showing predictions and true labels):")
    print(f"    - tsne_all_nodes.png")
    print(f"    - tsne_compounds.png") 
    print(f"    - tsne_proteins.png")
    print(f"    - tsne_pairs_predictions.png (model predictions)")
    print(f"    - tsne_pairs_true_labels.png (true labels)")
    print(f"    - score_distributions.png (classifier scores)")


def visualize_triplet_training_metrics(train_losses, val_losses, val_separations, val_pos_similarities, val_neg_similarities, results_dir):
    """Visualize triplet loss training metrics including separation and similarities"""
    print("  - Creating triplet training metrics visualization")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Triplet Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Separation over time
    ax2.plot(epochs, val_separations, 'g-', label='Separation', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Separation (Pos - Neg)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Similarities over time
    ax3.plot(epochs, val_pos_similarities, 'green', label='Positive Pairs', linewidth=2)
    ax3.plot(epochs, val_neg_similarities, 'red', label='Negative Pairs', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Similarity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-1, 1])
    
    # Separation histogram for last epoch
    if len(val_separations) > 0:
        last_separation = val_separations[-1]
        ax4.hist([val_pos_similarities[-1]], bins=20, alpha=0.6, label='Positive', color='green', density=True)
        ax4.hist([val_neg_similarities[-1]], bins=20, alpha=0.6, label='Negative', color='red', density=True)
        ax4.axvline(val_pos_similarities[-1], color='darkgreen', linestyle='--', linewidth=2)
        ax4.axvline(val_neg_similarities[-1], color='darkred', linestyle='--', linewidth=2)
        ax4.set_xlabel('Similarity')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "triplet_training_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Triplet training metrics saved to: {results_dir}/triplet_training_metrics.png")

def visualize_test_results_comprehensive(y_true, y_scores, y_pred, y_similarities, test_results, results_dir):
    """Create comprehensive test results visualization"""
    print("  - Creating comprehensive test results visualization")
    
    fig = plt.figure(figsize=(20, 15))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    pos_scores = [s for s, t in zip(y_scores, y_true) if t == 1]
    neg_scores = [s for s, t in zip(y_scores, y_true) if t == 0]
    ax1.hist(pos_scores, bins=30, alpha=0.6, label=f'Positive (n={len(pos_scores)})', color='green', density=True)
    ax1.hist(neg_scores, bins=30, alpha=0.6, label=f'Negative (n={len(neg_scores)})', color='red', density=True)
    ax1.axvline(test_results['best_threshold'], color='black', linestyle='--', label=f'Threshold: {test_results["best_threshold"]:.3f}')
    ax1.set_xlabel('Classification Score')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC curve
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    # 3. PR curve
    ax3 = fig.add_subplot(gs[0, 2])
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    ax3.plot(recall, precision, color='darkorange', lw=2, label=f'AP = {avg_precision:.3f}')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm = test_results['test_confusion_matrix']
    # FIX: Handle dict format of confusion matrix
    if isinstance(cm, dict):
        tp = cm['tp']
        tn = cm['tn']
        fp = cm['fp']
        fn = cm['fn']
    else:
        tp, tn, fp, fn = cm.ravel()
    cm_array = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])
    
    # 5. Metrics bar chart
    ax5 = fig.add_subplot(gs[1, 1:])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'Specificity', 'ROC-AUC']
    metrics_values = [
        test_results['test_accuracy'],
        test_results['test_precision'],
        test_results['test_recall'],
        test_results['test_f1'],
        test_results['test_mcc'],
        test_results['test_specificity'],
        test_results['test_roc_auc']
    ]
    bars = ax5.bar(
        metrics_names,
        metrics_values,
        color='#9EC5E5',
        edgecolor='#2F5D8A',
        linewidth=1.2,
    )
    ax5.set_ylim([0, 1.05])
    ax5.set_ylabel('Score')
    for bar, value in zip(bars, metrics_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Similarity vs Score scatter
    if y_similarities is not None:
        ax6 = fig.add_subplot(gs[2, :2])
        pos_mask = np.array(y_true) == 1
        neg_mask = np.array(y_true) == 0
        ax6.scatter(np.array(y_similarities)[pos_mask], np.array(y_scores)[pos_mask], 
                   c='green', alpha=0.6, label='Positive pairs', s=50)
        ax6.scatter(np.array(y_similarities)[neg_mask], np.array(y_scores)[neg_mask], 
                   c='red', alpha=0.6, label='Negative pairs', s=50)
        ax6.set_xlabel('Cosine Similarity')
        ax6.set_ylabel('Classification Score')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add correlation
        correlation = np.corrcoef(y_similarities, y_scores)[0, 1]
        ax6.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Performance summary text
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    # FIX: Handle dict format of confusion matrix in summary text
    if isinstance(cm, dict):
        tp_val = cm['tp']
        tn_val = cm['tn']
        fp_val = cm['fp']
        fn_val = cm['fn']
    else:
        tp_val, tn_val, fp_val, fn_val = cm.ravel()
        
    summary_text = f"""Performance Summary:
    
Balanced Accuracy: {test_results['test_accuracy']:.3f}
ROC-AUC: {test_results['test_roc_auc']:.3f}
F1-Score: {test_results['test_f1']:.3f}
MCC: {test_results['test_mcc']:.3f}

True Positives: {tp_val}
True Negatives: {tn_val}
False Positives: {fp_val}
False Negatives: {fn_val}

Sensitivity: {test_results['test_sensitivity']:.3f}
Specificity: {test_results['test_specificity']:.3f}

Best Threshold: {test_results['best_threshold']:.3f}
"""
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.savefig(os.path.join(results_dir, "test_results_comprehensive.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Comprehensive test results saved to: {results_dir}/test_results_comprehensive.png")

def create_additional_visualizations(train_losses, val_losses, val_aucs, test_results, results_dir="."):
    """Create additional visualizations for comprehensive analysis"""
    
    # training curves
    print("  - Creating training curves")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    # Check if val_aucs has data
    if len(val_aucs) > 0:
        # Create epochs for validation AUCs (they are computed every 5 epochs in classifier training)
        val_epochs = range(5, 5 + len(val_aucs) * 5, 5)[:len(val_aucs)]
        plt.plot(val_epochs, val_aucs, 'g-', label='Validation AUC', linewidth=2)
        plt.ylim([0, 1])
    else:
        # If no AUC data, show a placeholder
        plt.text(0.5, 0.5, 'No AUC data available', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12, color='gray')
    
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # create F1 progression (if available from classifier training)
    if hasattr(test_results, 'get') and 'val_f1s' in test_results:
        val_f1s = test_results['val_f1s']
        if len(val_f1s) > 0:
            val_epochs = range(5, 5 + len(val_f1s) * 5, 5)[:len(val_f1s)]
            plt.plot(val_epochs, val_f1s, 'purple', linewidth=2, label='Validation F1')
    
    plt.plot(epochs, [0.9] * len(epochs), 'purple', linewidth=2, linestyle='--', label='Target F1 (0.9)')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # metrics comparison
    print("  - Creating metrics comparison")
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']
    metrics_values = [
        test_results['test_accuracy'],
        test_results['test_precision'], 
        test_results['test_recall'],
        test_results['test_f1'],
        test_results['test_roc_auc'],
        test_results['test_mcc']
    ]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(
        metrics_names,
        metrics_values,
        color='#9EC5E5',
        edgecolor='#2F5D8A',
        linewidth=1.2,
    )
    plt.ylim([0, 1.05])
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    # add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

class ClassificationHead(nn.Module):
    """Classification head for phase 3 evaluation"""
    def __init__(self, input_dim, hidden_dim=128):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # *2 for concatenated embeddings
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, compound_emb, protein_emb):
        """Forward pass for classification"""
        pair_emb = torch.cat([compound_emb, protein_emb], dim=1)
        return self.classifier(pair_emb).squeeze()

def visualize_classifier_training(model, results_dir):
    """Visualize classifier training history with proper epoch alignment"""
    if not hasattr(model, 'classifier_training_history'):
        print("No classifier training history found")
        return
    
    history = model.classifier_training_history
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # FIX: Proper epoch alignment for validation metrics
    if len(history['val_aucs']) > 0:
        # Create epochs that match the validation data length
        val_epochs = range(5, 5 + len(history['val_aucs']) * 5, 5)
        ax2.plot(val_epochs, history['val_aucs'], 'g-', label='Validation AUC', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ROC-AUC')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    # Validation F1 with same epoch alignment
    if len(history['val_f1s']) > 0:
        val_epochs = range(5, 5 + len(history['val_f1s']) * 5, 5)
        ax3.plot(val_epochs, history['val_f1s'], 'purple', label='Validation F1', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
    
    # Learning rate
    if len(history['learning_rates']) > 0:
        ax4.plot(epochs, history['learning_rates'], 'orange', label='Learning Rate', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "classifier_training_progress.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Classifier training progress saved to: {results_dir}/classifier_training_progress.png")

def train_classifier_phase3(model, x, edge_index, edge_attr, node_types, train_pairs, val_pairs, data, 
                           num_epochs=50, lr=0.0001, device='cuda', results_dir='results'):
    """Phase 3: Train classification head with ADVANCED HARD NEGATIVE MINING"""
    print(f"\n=== PHASE 3: CLASSIFICATION HEAD TRAINING WITH HARD NEGATIVE MINING ({num_epochs} epochs) ===")
    print("Using advanced hard negative mining for better classifier training with STABLE OPTIMIZATION")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    model.eval()  # Keep GNN frozen
    model.pair_classifier.train()
    
    # Fixed learning rate optimizer (no scheduler)
    optimizer = torch.optim.Adam(
        model.pair_classifier.parameters(), 
        lr=lr, # Lower learning rate for classifier
        weight_decay=0 # No weight decay - removed L2 regularization
    )
    criterion = nn.BCELoss()
    
    # Get embeddings once (GNN is frozen)
    with torch.no_grad():
        embeddings, z = model(x, edge_index, edge_attr, node_types=node_types)
        compound_mask = node_types == 0
        protein_mask = node_types == 1
        z_compounds = z[compound_mask]
        z_proteins = z[protein_mask]
    
    # ADVANCED NEGATIVE SAMPLING with hard negative mining
    print("Pre-generating hard negatives using embedding similarity...")
    positive_set = set(train_pairs)
    num_compounds = data['num_compounds'] 
    num_proteins = data['num_proteins']
    
    def generate_hard_negatives(pairs, num_negatives, z_compounds, z_proteins):
        """Generate hard negatives using similarity-based mining"""
        hard_negatives = []
        
        # Create pair embeddings for positive pairs to compare against
        positive_pair_embeddings = []
        for c_idx, p_idx in pairs[:min(10, len(pairs))]:  # Use subset of positives for efficiency
            pair_emb = torch.cat([z_compounds[c_idx], z_proteins[p_idx]], dim=0)
            pair_emb = F.normalize(pair_emb, p=2, dim=0)
            positive_pair_embeddings.append(pair_emb)
        
        if len(positive_pair_embeddings) > 0:
            # Average positive pair embedding as reference
            reference_embedding = torch.stack(positive_pair_embeddings).mean(dim=0)
        else:
            # Fallback if no positive pairs
            reference_embedding = torch.zeros(z_compounds.size(1) + z_proteins.size(1), device=z_compounds.device)
        
        rng = np.random.RandomState(SEED)

        # Generate candidates and compute PAIR similarities
        all_candidates = []
        for _ in range(num_negatives * 3):  # Generate 3x candidates
            c_idx = rng.randint(0, num_compounds - 1)
            p_idx = rng.randint(0, num_proteins - 1)
            if (c_idx, p_idx) not in positive_set:
                # Create PAIR embedding for this candidate
                candidate_pair_emb = torch.cat([z_compounds[c_idx], z_proteins[p_idx]], dim=0)
                candidate_pair_emb = F.normalize(candidate_pair_emb, p=2, dim=0)
                
                # Compute similarity between PAIR embeddings
                sim = F.cosine_similarity(candidate_pair_emb.unsqueeze(0), reference_embedding.unsqueeze(0)).item()
                all_candidates.append(((c_idx, p_idx), sim))
        
        # Sort by similarity (descending) - high similarity = hard negative
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        hard_count = int(num_negatives * 0.7)  # 70% hard negatives
        random_count = num_negatives - hard_count  # 30% random negatives
        
        # Add hard negatives
        for i in range(min(hard_count, len(all_candidates))):
            hard_negatives.append(all_candidates[i][0])
        
        rng = np.random.RandomState(SEED)

        # Add some random negatives for diversity
        while len(hard_negatives) < num_negatives:
            c_idx = rng.randint(0, num_compounds - 1)
            p_idx = rng.randint(0, num_proteins - 1)
            if (c_idx, p_idx) not in positive_set and (c_idx, p_idx) not in hard_negatives:
                hard_negatives.append((c_idx, p_idx))
        
        return hard_negatives[:num_negatives]
    
    # Generate training and validation negatives
    train_negative_pairs = generate_hard_negatives(train_pairs, len(train_pairs), z_compounds, z_proteins)
    val_negative_pairs = generate_hard_negatives(val_pairs, len(val_pairs), z_compounds, z_proteins)
    
    print(f"Generated {len(train_negative_pairs)} hard training negatives")
    print(f"Generated {len(val_negative_pairs)} hard validation negatives")
    
    best_val_loss = float('inf')
    best_val_f1 = 0
    best_threshold = 0.5
    train_losses_clf = []
    val_losses_clf = []
    val_aucs_clf = []
    val_f1s_clf = []
    learning_rates = []
    patience = 0
    max_patience = 20
    eval_batch_size = 256
    
    for epoch in range(num_epochs):
        model.pair_classifier.train()
        
        # Training with hard negatives
        all_train_pairs = train_pairs + train_negative_pairs
        train_labels = [1] * len(train_pairs) + [0] * len(train_negative_pairs)
        
        # Shuffle training data
        combined = list(zip(all_train_pairs, train_labels))
        random.shuffle(combined)
        all_train_pairs, train_labels = zip(*combined)
        
        train_loss = 0
        num_batches = 0
        batch_size = 64
        
        for i in range(0, len(all_train_pairs), batch_size):
            batch_end = min(i + batch_size, len(all_train_pairs))
            batch_pairs = all_train_pairs[i:batch_end]
            batch_labels = torch.tensor(train_labels[i:batch_end], dtype=torch.float32, device=device)
            
            # Extract compound and protein indices from batch_pairs
            batch_compound_indices = torch.tensor([pair[0] for pair in batch_pairs], device=device)
            batch_protein_indices = torch.tensor([pair[1] for pair in batch_pairs], device=device)
            
            optimizer.zero_grad()
            logits, _ = model.classify_pairs(z_compounds, z_proteins, batch_compound_indices, batch_protein_indices)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_losses_clf.append(avg_train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Validation WITHOUT regularization
        model.pair_classifier.eval()
        with torch.no_grad():
            val_loss = 0
            all_val_pairs = val_pairs + val_negative_pairs
            val_labels = torch.tensor([1] * len(val_pairs) + [0] * len(val_negative_pairs), dtype=torch.float32, device=device)
            
            # Process validation in batches
            for i in range(0, len(all_val_pairs), eval_batch_size):
                batch_end = min(i + eval_batch_size, len(all_val_pairs))
                batch_pairs = all_val_pairs[i:batch_end]
                batch_labels = val_labels[i:batch_end]
                
                # Extract compound and protein indices
                batch_compound_indices = torch.tensor([pair[0] for pair in batch_pairs], device=device)
                batch_protein_indices = torch.tensor([pair[1] for pair in batch_pairs], device=device)
                
                logits, probabilities = model.classify_pairs(z_compounds, z_proteins, batch_compound_indices, batch_protein_indices)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item() * len(batch_pairs)
            
            avg_val_loss = val_loss / len(all_val_pairs)
            val_losses_clf.append(avg_val_loss)
            
            # Compute validation metrics
            val_compound_indices = torch.tensor([pair[0] for pair in all_val_pairs], device=device)
            val_protein_indices = torch.tensor([pair[1] for pair in all_val_pairs], device=device)
            
            _, val_probs = model.classify_pairs(z_compounds, z_proteins, val_compound_indices, val_protein_indices)
            val_probs_cpu = val_probs.cpu().numpy()
            val_labels_cpu = val_labels.cpu().numpy()
            
            # Find optimal threshold and calculate metrics
            optimal_threshold, _ = find_optimal_threshold_balanced(val_labels_cpu, val_probs_cpu, metric='f1')
            val_pred = (val_probs_cpu > optimal_threshold).astype(int)
            
            val_auc = roc_auc_score(val_labels_cpu, val_probs_cpu)
            val_f1 = f1_score(val_labels_cpu, val_pred)
            
            val_aucs_clf.append(val_auc)
            val_f1s_clf.append(val_f1)
            
            if avg_val_loss < best_val_loss and val_f1 > best_val_f1 * 0.95:
                best_val_loss = avg_val_loss
                best_val_f1 = val_f1
                best_threshold = optimal_threshold
                patience = 0
            else:
                patience += 1
        
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                  f"Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}, Val<Train: {'✓' if avg_val_loss < avg_train_loss else '✗'}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            # Load best classifier
            try:
                model.pair_classifier.load_state_dict(torch.load(os.path.join(results_dir, 'best_classifier.pth')))
            except Exception as e:
                print(f"  Could not load best classifier: {e}")
            break
    
    # Store training history
    model.classifier_training_history = {
        'train_losses': train_losses_clf,
        'val_losses': val_losses_clf,
        'val_aucs': val_aucs_clf,
        'val_f1s': val_f1s_clf,
        'learning_rates': learning_rates,
        'hard_negative_mining': True
    }
    
    print(f"  Best validation Loss: {best_val_loss:.4f}, F1: {best_val_f1:.4f}, Threshold: {best_threshold:.3f}")
    print(f"  Classifier training completed with stable optimization")
    
    return model, best_threshold

def train_model_3_phase(model, data, num_epochs_phase1=100, num_epochs_phase2=100, num_epochs_phase3=30, 
                       lr=0.0001, test_ratio=0.15, val_ratio=0.15, results_dir="results"):
    """3-phase training with ADVANCED HARD NEGATIVE MINING throughout"""
    
    # GPU setup with memory optimization
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        
        # Enable automatic mixed precision for faster training
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
        print("Automatic Mixed Precision (AMP) enabled for faster training")
    else:
        device = torch.device('cpu')
        print("Running on CPU - Training will be significantly slower!")
        use_amp = False
        scaler = None

    model = model.to(device)
    
    # MANDATORY: Ensure model has edge masking enabled
    if not model.use_edge_mask:
        print("WARNING: Model was created without edge masking. Enabling edge masking now...")
        model.use_edge_mask = True
        # Initialize edge mask if not present
        if not hasattr(model, 'edge_mask') or model.edge_mask is None:
            # Get intermediate dimension from model layers
            intermediate_dim = model.convs[-2].lin.out_features if len(model.convs) > 1 else model.convs[0].lin.out_features
            model.edge_mask = SoftEdgeMask(
                compound_dim=intermediate_dim,
                protein_dim=intermediate_dim,
                hidden_dim=128,
                num_heads=4,
                top_k_ratio=0.5
            ).to(device)
    print(f"Edge masking status: {'ENABLED' if model.use_edge_mask else 'DISABLED'}")
    assert model.use_edge_mask, "Edge masking must be enabled for 3-phase training!"
    
    # Load data to GPU
    x = torch.FloatTensor(data['node_features']).to(device)
    edge_index = torch.LongTensor(data['edge_index']).to(device)
    
    if 'edge_attr' in data and data['edge_attr'].size > 0:
        edge_attr = torch.FloatTensor(data['edge_attr']).to(device)
    else:
        edge_attr = None
    
    node_types = data['node_types']
    node_types_gpu = torch.from_numpy(node_types).to(device)
    
    # Prepare data_gpu for edge masking
    data_gpu = {
        'num_compounds': data['num_compounds'],
        'num_proteins': data['num_proteins']
    }
    if 'cp_edge_index' in data:
        data_gpu['cp_edge_index'] = torch.LongTensor(data['cp_edge_index']).to(device)
    
    # Split data ONCE and use consistently across all phases
    train_pairs, val_pairs, test_pairs = split_interaction_data(data, test_ratio=test_ratio, val_ratio=val_ratio)
    print(f"\nData splits (used consistently across all phases):")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Val: {len(val_pairs)} pairs") 
    print(f"  Test: {len(test_pairs)} pairs")
    
    # ENHANCED TRIPLET LOSS with advanced hard negative mining
    criterion = TripletContrastiveLoss(
        margin=2.0,  # Reduced margin for stability
        max_negatives_per_positive=10,  # Reduced for balance
        batch_size=64,  # Smaller batch for better gradients
        temperature=1.0,  # Slightly increased temperature
        pos_weight=1.0,
        neg_weight=1.0,  # Balanced weights
        hard_negative_mining=True
    )
    
    # Fixed learning rate optimizer (no scheduler)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0,  # No weight decay - removed L2 regularization
        betas=(0.9, 0.999)
    )
    
    print(f"\nENHANCED 3-PHASE TRAINING")
    print(f"  - Fixed learning rate: {lr}")
    print(f"  - Conservative triplet loss only")
    print(f"  - No noise augmentation")
    print(f"  - No L2 regularization")
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    val_separations = []
    val_pos_similarities = []
    val_neg_similarities = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # PHASE 1: Standard triplet loss training
    print(f"\n=== PHASE 1: STANDARD TRIPLET LOSS ({num_epochs_phase1} epochs) ===")
    
    for epoch in range(num_epochs_phase1):
        model.train()
        optimizer.zero_grad()
        
        # Training with model in train mode (dropout active)
        embeddings, z = model(x, edge_index, edge_attr, node_types=node_types)
        
        compound_mask = node_types == 0
        protein_mask = node_types == 1
        
        z_compounds = z[compound_mask]
        z_proteins = z[protein_mask]
        
        # Moderate margin for phase 1
        criterion.margin = 2.0

        # Training loss without regularization
        loss = criterion(z_compounds, z_proteins, train_pairs, apply_regularization=False)
        
        # No L2 penalty
        total_loss = loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation: Set model to eval mode (disables dropout automatically)
        model.eval()
        with torch.no_grad():
            embeddings, z = model(x, edge_index, edge_attr, node_types=node_types)
            z_compounds = z[compound_mask]
            z_proteins = z[protein_mask]
            # Clean validation - NO regularization
            val_loss = criterion(z_compounds, z_proteins, val_pairs, apply_regularization=False)
            
            # --- IMPROVED SEPARATION CALCULATION ---
            def get_pair_emb(c_idx, p_idx):
                return criterion.create_pair_embedding(
                    z_compounds[c_idx].unsqueeze(0),
                    z_proteins[p_idx].unsqueeze(0)
                ).squeeze(0)
            pos_pair_embs = []
            neg_pair_embs = []
            sample_size = min(100, len(val_pairs))
            pos_sample = random.sample(val_pairs, sample_size)
            for c_idx, p_idx in pos_sample:
                pos_pair_embs.append(get_pair_emb(c_idx, p_idx))
            positive_set = set(val_pairs)
            rng = np.random.RandomState(SEED)
            neg_count = 0
            while neg_count < sample_size:
                c_idx = rng.randint(0, data['num_compounds'] - 1)
                p_idx = rng.randint(0, data['num_proteins'] - 1)
                if (c_idx, p_idx) not in positive_set:
                    neg_pair_embs.append(get_pair_emb(c_idx, p_idx))
                    neg_count += 1
            pos_pair_embs = torch.stack(pos_pair_embs)
            neg_pair_embs = torch.stack(neg_pair_embs)
            # Use cosine similarity for separation
            pos_sim = F.cosine_similarity(pos_pair_embs, pos_pair_embs.mean(dim=0, keepdim=True)).mean().item()
            neg_sim = F.cosine_similarity(neg_pair_embs, pos_pair_embs.mean(dim=0, keepdim=True)).mean().item()
            separation = pos_sim - neg_sim

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            val_separations.append(separation)
            val_pos_similarities.append(pos_sim)
            val_neg_similarities.append(neg_sim)
            
            # No scheduler - fixed learning rate
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save best model with proper state dict
                torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_phase1.pth'))
            else:
                patience_counter += 1
        
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs_phase1}: "
                  f"Train={loss.item():.4f}, Val={val_loss.item():.4f}, "
                  f"Sep={separation:.4f}, Val<Train: {'✓' if val_loss.item() < loss.item() else '✗'}")
        
        if patience_counter >= 80:  # Increased patience
            print(f"Early stopping at epoch {epoch+1} (patience: {patience_counter})")
            try:
                model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model_phase1.pth')))
            except Exception as e:
                print(f"Could not load best model: {e}")
            break

    print(f"\n=== PHASE 2: LEARNING GENERALIZABLE EDGE MASKING ({num_epochs_phase2} epochs) ===")
    print("Extended phase 2 with stable training and no noise")
    
    # Keep same optimizer with fixed learning rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,  # Slightly reduced for phase 2
        weight_decay=0 # No weight decay
    )
    
    patience_counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs_phase2):
        model.train()
        
        # NO NOISE - clean inputs
        optimizer.zero_grad()
        
        data_gpu['epoch'] = epoch
        
        # Apply edge masking with AMP if available
        if use_amp and torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                embeddings, z = model(x, edge_index, edge_attr, node_types=node_types_gpu, 
                                     apply_edge_mask=True, data=data_gpu)
                
                compound_mask = node_types == 0
                protein_mask = node_types == 1
                
                z_compounds = z[compound_mask]
                z_proteins = z[protein_mask]
                
                # Light dropout only (no noise)
                z_compounds_aug = z_compounds
                z_proteins_aug = z_proteins
                
                loss = criterion(z_compounds_aug, z_proteins_aug, train_pairs, apply_regularization=False)
                
                # No L2 penalty or regularization
                edge_mask_reg = 1e-4 * sum(p.pow(2).sum() for p in model.edge_mask.parameters()) if hasattr(model, 'edge_mask') else 0
                total_loss = loss + edge_mask_reg
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without AMP
            embeddings, z = model(x, edge_index, edge_attr, node_types=node_types_gpu, 
                                  apply_edge_mask=True, data=data_gpu)
            
            compound_mask = node_types == 0
            protein_mask = node_types == 1
            
            z_compounds = z[compound_mask]
            z_proteins = z[protein_mask]
            
            # Light dropout only (no noise)
            z_compounds_aug = z_compounds
            z_proteins_aug = z_proteins
            
            loss = criterion(z_compounds_aug, z_proteins_aug, train_pairs, apply_regularization=False)
            
            # No L2 penalty
            edge_mask_reg = 1e-4 * sum(p.pow(2).sum() for p in model.edge_mask.parameters()) if hasattr(model, 'edge_mask') else 0
            total_loss = loss + edge_mask_reg
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation with edge masking - NO regularization
        model.eval()
        with torch.no_grad():
            val_data_gpu = data_gpu.copy()
            val_data_gpu['epoch'] = epoch
            # Clean validation without dropout or regularization
            embeddings, z = model(x, edge_index, edge_attr, node_types=node_types_gpu,
                                  apply_edge_mask=True, data=val_data_gpu)
            z_compounds = z[compound_mask]
            z_proteins = z[protein_mask]
            val_loss = criterion(z_compounds, z_proteins, val_pairs, apply_regularization=False)
            
            # --- FIXED SEPARATION CALCULATION ---
            def get_pair_emb(c_idx, p_idx):
                return criterion.create_pair_embedding(
                    z_compounds[c_idx].unsqueeze(0),
                    z_proteins[p_idx].unsqueeze(0)
                ).squeeze(0)
            pos_pair_embs = []
            neg_pair_embs = []
            sample_size = min(100, len(val_pairs))
            pos_sample = random.sample(val_pairs, sample_size)
            for c_idx, p_idx in pos_sample:
                pos_pair_embs.append(get_pair_emb(c_idx, p_idx))
            positive_set = set(val_pairs)
            neg_count = 0
            rng = np.random.RandomState(SEED)
            while neg_count < sample_size:
                c_idx = rng.randint(0, data['num_compounds'] - 1)
                p_idx = rng.randint(0, data['num_proteins'] - 1)
                if (c_idx, p_idx) not in positive_set:
                    neg_pair_embs.append(get_pair_emb(c_idx, p_idx))
                    neg_count += 1
            pos_pair_embs = torch.stack(pos_pair_embs)
            neg_pair_embs = torch.stack(neg_pair_embs)
            # Use cosine similarity for separation
            pos_sim = F.cosine_similarity(pos_pair_embs, pos_pair_embs.mean(dim=0, keepdim=True)).mean().item()
            neg_sim = F.cosine_similarity(neg_pair_embs, pos_pair_embs.mean(dim=0, keepdim=True)).mean().item()
            separation = pos_sim - neg_sim

            train_losses.append(total_loss.item())  # Regularized training loss
            val_losses.append(val_loss.item())  # Clean validation loss
            val_separations.append(separation)
            val_pos_similarities.append(pos_sim)
            val_neg_similarities.append(neg_sim)
            
            # Check for overfitting but be less strict
            if val_loss.item() > total_loss.item() * 1.5:
                print(f"  Warning: Possible overfitting (Val loss {val_loss.item():.4f} > Train loss {total_loss.item():.4f} * 1.5)")
            if separation < 0.05:
                print(f"  Warning: Low separation ({separation:.4f})")
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save best model with proper state dict
                torch.save(model.state_dict(), os.path.join(results_dir, 'best_model_phase2.pth'))
            else:
                patience_counter += 1
        
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs_phase2}: "
                  f"Train={total_loss.item():.4f}, Val={val_loss.item():.4f}, "
                  f"Sep={separation:.4f}, Val<Train: {'✓' if val_loss.item() < total_loss.item() else '✗'}")
            # Show edge masking progress
            if hasattr(model, 'edge_mask') and model.edge_mask.last_selected_pairs is not None:
                print(f"  Edge mask: {model.edge_mask.last_selected_pairs.size(1)} selected edges")

        # Early stopping with longer patience for phase 2
        if patience_counter >= 200:  # Increased patience for stability
            print(f"Early stopping at epoch {epoch+1} (patience: {patience_counter})")
            try:
                model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model_phase2.pth')))
            except Exception as e:
                print(f"Could not load best model: {e}")
            break
    
    # PHASE 3: Classification with same overfitting prevention
    print(f"\n=== PHASE 3: CLASSIFICATION WITH OVERFITTING PREVENTION ({num_epochs_phase3} epochs) ===")
    
    trained_model, best_threshold = train_classifier_phase3(
        model, x, edge_index, edge_attr, node_types, 
        train_pairs, val_pairs, data, 
        num_epochs=num_epochs_phase3,
        lr=lr,
        device=device,
        results_dir=results_dir
    )
    
    # Extract classifier training metrics for visualization
    val_aucs_clf = []
    if hasattr(trained_model, 'classifier_training_history'):
        val_aucs_clf = trained_model.classifier_training_history.get('val_aucs', [])
        
    # Visualize classifier training
    visualize_classifier_training(trained_model, results_dir)
    
    # Final evaluation with classifier
    print("\n=== FINAL EVALUATION: TESTING LEARNED EDGE MASKING ON TEST GRAPH ===")
    print("This demonstrates the model's ability to improve results on NEW, UNSEEN graphs")
    
    y_true, y_scores, y_similarities = evaluate_with_classifier(
        trained_model, x, edge_index, edge_attr, node_types_gpu, 
        test_pairs, data, threshold=best_threshold
    )
    y_pred = [1 if score > best_threshold else 0 for score in y_scores]

    # Analyze misclassified true positives (false negatives)
    print("\n=== ANALYZING MISCLASSIFIED INTERACTIONS ===")
    fn_report = analyze_misclassified_interactions(y_true, y_pred, test_pairs, data, results_dir)
        
    # Calculate comprehensive metrics
    test_metrics = calculate_balanced_metrics(y_true, y_pred, y_scores)
    
    # Create final results
    test_results = {
        'test_pairs': test_pairs,
        'test_accuracy': test_metrics['balanced_accuracy'], 
        'test_balanced_accuracy': test_metrics['balanced_accuracy'], 
        'test_regular_accuracy': test_metrics['accuracy'], 
        'test_roc_auc': test_metrics['roc_auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1_score'],
        'test_mcc': test_metrics['mcc'],
        'test_ap': test_metrics['average_precision'],
        'test_sensitivity': test_metrics['sensitivity'],
        'test_specificity': test_metrics['specificity'],
        'test_tpr': test_metrics['sensitivity'],
        'test_tnr': test_metrics['specificity'],
        'test_fpr': 1 - test_metrics['specificity'],
        'test_fnr': 1 - test_metrics['sensitivity'],
        'test_confusion_matrix': test_metrics['confusion_matrix'],
        'best_threshold': best_threshold,
        'evaluation_method': '3phase_classification_head_with_mandatory_edge_masking'
    }
    
    # Print results
    print(f"\n=== FINAL TEST RESULTS ===")
    print(f"Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"ROC-AUC: {test_results['test_roc_auc']:.4f}")
    print(f"Precision: {test_results['test_precision']:.4f}")
    print(f"Recall: {test_results['test_recall']:.4f}")
    print(f"F1-Score: {test_results['test_f1']:.4f}")
    print(f"MCC: {test_results['test_mcc']:.4f}")
    
    # Save model and results
    print(f"\nSaving model and results...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'embeddings': embeddings,
        'test_results': test_results,
        'training_completed': True
    }, os.path.join(results_dir, 'trained_model.pth'))
    
    # Convert test results to JSON-serializable format
    test_results_json = convert_numpy_to_python(test_results)
    
    # Save test results as JSON
    with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results_json, f, indent=2)
    
    print(f"Model saved to: {results_dir}/trained_model.pth")
    print(f"Test results saved to: {results_dir}/test_results.json")
    print(f"All visualizations saved to: {results_dir}/")
    
    print(f"\n{'='*80}")
    print("Training and evaluation complete!")
    print(f"{'='*80}")

    print("\n=== SOFT EDGE VALIDATION: Testing if model identifies real interactions ===")
    validation_results = validate_soft_edges_on_test_data(
        trained_model, x, edge_index, edge_attr, node_types_gpu,
        test_pairs, data, results_dir
    )

    print(f"\nKey Finding: Model achieves {validation_results['average_precision']:.3f} average precision")
    print(f"This is {validation_results['improvement_factor']:.1f}x better than random!")

    print("\n=== CREATING VISUALIZATIONS ===")
    
    # 1. Triplet training metrics visualization
    visualize_triplet_training_metrics(
        train_losses, val_losses, val_separations, 
        val_pos_similarities, val_neg_similarities, results_dir
    )
    
    # 2. Score distributions
    visualize_score_distributions(
        y_true, y_scores, epoch=None, 
        save_path=os.path.join(results_dir, "final_score_distributions.png")
    )
    
    # 3. ROC and PR curves
    visualize_roc_pr_curves(
        y_true, y_scores, 
        save_path=os.path.join(results_dir, "roc_pr_curves.png")
    )
    
    # 4. Confusion matrix
    visualize_confusion_matrix(
        y_true, y_pred, 
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )
    
    # 5. Comprehensive t-SNE visualizations
    print("\n=== CREATING t-SNE VISUALIZATIONS ===")
    visualize_embeddings_tsne_comprehensive(
        embeddings.cpu().numpy(), 
        node_types, 
        y_true, 
        y_scores, 
        y_pred, 
        test_pairs,
        z_compounds.cpu().detach(),
        z_proteins.cpu().detach(),
        results_dir=results_dir,
        best_threshold=best_threshold
    )
    
    # 6. Test results comprehensive visualization
    visualize_test_results_comprehensive(
        y_true, y_scores, y_pred, y_similarities, 
        test_results, results_dir
    )
    
    # 7. Additional training curves
    create_additional_visualizations(
        train_losses, val_losses, val_aucs_clf,  # Now passing the actual classifier AUCs
        test_results, results_dir
    )
    
    # 8. Attention maps visualization (if edge masking is enabled)
    if model.use_edge_mask:
        visualize_attention_maps(
            model, x, edge_index, edge_attr, node_types_gpu, 
            data_gpu, results_dir, final_epoch=num_epochs_phase2
        )
    
    print(f"\n=== ALL VISUALIZATIONS COMPLETED ===")
    print(f"Visualizations saved to: {results_dir}/")
    print(f"  - triplet_training_metrics.png")
    print(f"  - final_score_distributions.png")
    print(f"  - roc_pr_curves.png")
    print(f"  - confusion_matrix.png")
    print(f"  - tsne_all_nodes.png")
    print(f"  - tsne_compounds.png")
    print(f"  - tsne_proteins.png")
    print(f"  - tsne_pairs_predictions.png")
    print(f"  - tsne_pairs_true_labels.png")
    print(f"  - test_results_comprehensive.png")
    print(f"  - training_curves.png")
    print(f"  - metrics_comparison.png")
    print(f"  - classifier_training_progress.png")

    if model.use_edge_mask:
        print(f"  - full_attention_with_selected_highlighted.png")
        print(f"  - attention_comparison_side_by_side.png")
        print(f"  - attention_score_distribution_with_selection.png")
        print(f"  - selected_edges_heatmap.png")
        print(f"  - selected_edges_scatter.png")
    
    return trained_model, embeddings, test_results

def evaluate_with_classifier(model, x, edge_index, edge_attr, node_types, test_pairs, data, threshold=0.5):
    """
    Evaluation using the learned classifier head instead of cosine similarity threshold
    
    This is more sophisticated than simple threshold-based classification as it can learn
    non-linear decision boundaries.
    """
    model.eval()
    device = x.device if hasattr(x, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        # Convert node_types to GPU tensor if it's numpy array
        if isinstance(node_types, np.ndarray):
            node_types_gpu = torch.from_numpy(node_types).to(device)
        else:
            node_types_gpu = node_types
            
        embeddings, z = model(x, edge_index, edge_attr, node_types=node_types_gpu)
        
        compound_mask = node_types_gpu == 0
        protein_mask = node_types_gpu == 1
        
        z_compounds = z[compound_mask]
        z_proteins = z[protein_mask]
        
        # Normalize embeddings for similarity computation
        z_compounds_norm = F.normalize(z_compounds, p=2, dim=1)
        z_proteins_norm = F.normalize(z_proteins, p=2, dim=1)
        
        y_true = []
        y_scores = []
        y_similarities = []
        
        # Evaluate positive pairs
        pos_compound_indices = torch.tensor([c_idx for c_idx, p_idx in test_pairs], device=z_compounds.device)
        pos_protein_indices = torch.tensor([p_idx for c_idx, p_idx in test_pairs], device=z_proteins.device)
        
        logits, _ = model.classify_pairs(z_compounds, z_proteins, pos_compound_indices, pos_protein_indices)
        pos_probs = torch.sigmoid(logits)
        
        for i, prob in enumerate(pos_probs):
            y_true.append(1)
            y_scores.append(prob.item())
            
            # Also compute cosine similarity for comparison
            c_idx, p_idx = test_pairs[i]
            similarity = F.cosine_similarity(
                z_compounds_norm[c_idx].unsqueeze(0),
                z_proteins_norm[p_idx].unsqueeze(0)
            ).item()
            y_similarities.append(similarity)
        
        # Create and evaluate negative pairs
        num_compounds = data['num_compounds']
        num_proteins = data['num_proteins']
        positive_set = set(test_pairs)
        negative_pairs = []
        attempts = 0

        rng = np.random.RandomState(SEED)
        
        while len(negative_pairs) < len(test_pairs) and attempts < len(test_pairs) * 10:
            c_idx = rng.randint(0, num_compounds - 1)
            p_idx = rng.randint(0, num_proteins - 1)
            if (c_idx, p_idx) not in positive_set:
                negative_pairs.append((c_idx, p_idx))
        
        # Evaluate negative pairs in batches
        neg_compound_indices = torch.tensor([c_idx for c_idx, p_idx in negative_pairs], device=z_compounds.device)
        neg_protein_indices = torch.tensor([p_idx for c_idx, p_idx in negative_pairs], device=z_proteins.device)
        
        neg_logits, _ = model.classify_pairs(z_compounds, z_proteins, neg_compound_indices, neg_protein_indices)
        neg_probs = torch.sigmoid(neg_logits)
        
        for i, prob in enumerate(neg_probs):
            y_true.append(0)
            y_scores.append(prob.item())
            
            # Compute cosine similarity for comparison
            c_idx, p_idx = negative_pairs[i]
            similarity = F.cosine_similarity(
                z_compounds_norm[c_idx].unsqueeze(0),
                z_proteins_norm[p_idx].unsqueeze(0)
            ).item()
            y_similarities.append(similarity)
    
    return y_true, y_scores, y_similarities  # Return both classifier scores and similarities

def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj
    
def visualize_attention_maps(model, x, edge_index, edge_attr, node_types, data, results_dir, final_epoch=0):
    """Visualize attention score distribution and selected edges scatter plot only"""
    print("  - Creating attention map visualizations (distribution + scatter only)")
    if not model.use_edge_mask or not hasattr(model, 'edge_mask'):
        print("    Edge masking not enabled or edge mask module not found")
        return
    
    model.eval()
    with torch.no_grad():
        # Apply GNN layers to get embeddings
        compound_mask = node_types == 0
        protein_mask = node_types == 1
        
        # Process through GNN layers
        x_current = x
        for i, (conv, bn) in enumerate(zip(model.convs[:-1], model.bns)):
            x_current = conv(x_current, edge_index, edge_attr)
            x_current = bn(x_current)
            x_current = F.relu(x_current)
        
        intermediate_x = x_current
        
        # Extract embeddings
        compound_embeddings = intermediate_x[compound_mask]
        protein_embeddings = intermediate_x[protein_mask]
        num_compounds = compound_embeddings.size(0)
        num_proteins = protein_embeddings.size(0)
        
        # Get FULL attention scores for C-P (compound-protein)
        full_attention_scores = model.edge_mask.compute_attention_scores(
            compound_embeddings, protein_embeddings
        )
        
        # Get FULL attention scores for C-C (compound-compound)
        cc_attention_scores = None
        if hasattr(model, 'compound_edge_mask') and num_compounds > 1:
            cc_attention_scores = model.compound_edge_mask.compute_self_attention_scores(compound_embeddings)
        
        # Get FULL attention scores for P-P (protein-protein)
        pp_attention_scores = None
        if hasattr(model, 'protein_edge_mask') and num_proteins > 1:
            pp_attention_scores = model.protein_edge_mask.compute_self_attention_scores(protein_embeddings)
        
        # Get selected edges and their weights from the edge mask
        data['epoch'] = final_epoch
        edge_indices, edge_weights = model.edge_mask(
            compound_embeddings, protein_embeddings, 
            None, node_types, num_compounds, num_proteins, 
            epoch=final_epoch
        )
        
        # Convert to numpy for visualization
        full_attention_matrix = full_attention_scores.cpu().numpy()
        
        # Create binary mask for selected edges
        selected_mask = np.zeros_like(full_attention_matrix, dtype=bool)
        if edge_indices.size(1) > 0:
            # Extract compound-protein pairs (first half of bidirectional edges)
            num_unidirectional = edge_indices.size(1) // 2
            for i in range(num_unidirectional):
                c_idx = edge_indices[0, i].item()
                p_idx = edge_indices[1, i].item() - num_compounds
                if 0 <= c_idx < num_compounds and 0 <= p_idx < num_proteins:
                    selected_mask[c_idx, p_idx] = True
        
        # ========== ORIGINAL C-P ATTENTION SCORE DISTRIBUTION PLOT ==========
        print("    Creating C-P attention score distribution")
        
        # Extract scores
        all_scores = full_attention_matrix.flatten()
        selected_scores = full_attention_matrix[selected_mask]
        non_selected_scores = full_attention_matrix[~selected_mask]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Histogram of attention scores
        bins = np.linspace(0, all_scores.max(), 50)
        
        # Plot all scores
        ax1.hist(all_scores, bins=bins, alpha=0.5, label=f'All edges (n={len(all_scores)})', 
                 color='gray', edgecolor='black')
        
        # Overlay selected scores
        if len(selected_scores) > 0:
            ax1.hist(selected_scores, bins=bins, alpha=0.8, 
                     label=f'Selected edges (n={len(selected_scores)})', 
                     color='red', edgecolor='darkred')
        
        # Find and plot the effective threshold
        if len(selected_scores) > 0 and len(non_selected_scores) > 0:
            threshold = min(selected_scores)
            ax1.axvline(threshold, color='green', linestyle='--', linewidth=2, 
                       label=f'Threshold: {threshold:.4f}')
        
        ax1.set_xlabel('Attention Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = (f"Total edges: {len(all_scores)}\n"
                     f"Selected: {len(selected_scores)} ({len(selected_scores)/len(all_scores)*100:.2f}%)\n"
                     f"Mean (all): {all_scores.mean():.4f}\n"
                     f"Mean (selected): {selected_scores.mean():.4f}" if len(selected_scores) > 0 else "")
        ax1.text(0.65, 0.95, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Scatter plot of selected edges
        selected_positions = np.where(selected_mask)
        
        if len(selected_positions[0]) > 0:
            # Create sparse attention matrix with only selected edges
            attention_matrix = np.zeros((num_compounds, num_proteins))
            attention_matrix[selected_mask] = full_attention_matrix[selected_mask]
            
            # Extract non-zero scores for selected edges
            c_indices, p_indices = selected_positions
            scores = full_attention_matrix[selected_mask]
            
            # Create scatter plot
            scatter = ax2.scatter(p_indices, c_indices, c=scores, cmap='hot', 
                                 s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax2.set_xlabel('Protein Index', fontsize=12)
            ax2.set_ylabel('Compound Index', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax2, label='Attention Score')
            
            # Set axis limits with padding
            ax2.set_xlim(-1, num_proteins)
            ax2.set_ylim(-1, num_compounds)
            
            # Add stats
            stats_text2 = (f"Score range: [{scores.min():.4f}, {scores.max():.4f}]\n"
                          f"Mean score: {scores.mean():.4f}")
            ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No edges selected', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14, color='gray')
            ax2.set_xlabel('Protein Index', fontsize=12)
            ax2.set_ylabel('Compound Index', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "attention_score_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    C-P attention visualization saved to: {results_dir}/attention_score_distribution.png")
        
        # ========== NEW: C-C AND P-P DISTRIBUTIONS ==========
        print("    Creating C-C and P-P attention score distributions")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. C-P Distribution (repeat for comparison)
        ax = axes[0]
        ax.hist(all_scores, bins=bins, alpha=0.5, label=f'All C-P edges (n={len(all_scores)})', 
                color='gray', edgecolor='black')
        if len(selected_scores) > 0:
            ax.hist(selected_scores, bins=bins, alpha=0.8, 
                    label=f'Selected C-P edges (n={len(selected_scores)})', 
                    color='red', edgecolor='darkred')
        ax.set_xlabel('Attention Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Compound-Protein (C-P)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. C-C Distribution
        ax = axes[1]
        if cc_attention_scores is not None:
            cc_matrix = cc_attention_scores.cpu().numpy()
            # Get upper triangular to avoid duplicates and diagonal
            cc_triu = np.triu(cc_matrix, k=1)
            cc_scores = cc_triu[cc_triu > 0]
            
            if len(cc_scores) > 0:
                bins_cc = np.linspace(0, cc_scores.max(), 50)
                ax.hist(cc_scores, bins=bins_cc, alpha=0.7, 
                        label=f'C-C edges (n={len(cc_scores)})', 
                        color='blue', edgecolor='darkblue')
                
                # Add statistics
                stats_text = (f"Total edges: {len(cc_scores)}\n"
                             f"Mean: {cc_scores.mean():.4f}\n"
                             f"Std: {cc_scores.std():.4f}\n"
                             f"Max: {cc_scores.max():.4f}")
                ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, 
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No C-C edges computed', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12, color='gray')
        else:
            ax.text(0.5, 0.5, 'C-C edge masking not available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, color='gray')
        
        ax.set_xlabel('Attention Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Compound-Compound (C-C)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. P-P Distribution
        ax = axes[2]
        if pp_attention_scores is not None:
            pp_matrix = pp_attention_scores.cpu().numpy()
            # Get upper triangular to avoid duplicates and diagonal
            pp_triu = np.triu(pp_matrix, k=1)
            pp_scores = pp_triu[pp_triu > 0]
            
            if len(pp_scores) > 0:
                bins_pp = np.linspace(0, pp_scores.max(), 50)
                ax.hist(pp_scores, bins=bins_pp, alpha=0.7, 
                        label=f'P-P edges (n={len(pp_scores)})', 
                        color='green', edgecolor='darkgreen')
                
                # Add statistics
                stats_text = (f"Total edges: {len(pp_scores)}\n"
                             f"Mean: {pp_scores.mean():.4f}\n"
                             f"Std: {pp_scores.std():.4f}\n"
                             f"Max: {pp_scores.max():.4f}")
                ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, 
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No P-P edges computed', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=12, color='gray')
        else:
            ax.text(0.5, 0.5, 'P-P edge masking not available', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12, color='gray')
        
        ax.set_xlabel('Attention Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Protein-Protein (P-P)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Attention Score Distributions by Edge Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "attention_distributions_all_types.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    All edge type distributions saved to: {results_dir}/attention_distributions_all_types.png")
        
        # ========== OVERLAY COMPARISON PLOT ==========
        print("    Creating overlaid comparison plot")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare all scores with labels
        all_distributions = []
        
        # C-P scores
        all_distributions.append(('C-P (Compound-Protein)', all_scores, 'red', 0.5))
        
        # C-C scores
        if cc_attention_scores is not None:
            cc_matrix = cc_attention_scores.cpu().numpy()
            cc_triu = np.triu(cc_matrix, k=1)
            cc_scores = cc_triu[cc_triu > 0]
            if len(cc_scores) > 0:
                all_distributions.append(('C-C (Compound-Compound)', cc_scores, 'blue', 0.5))
        
        # P-P scores
        if pp_attention_scores is not None:
            pp_matrix = pp_attention_scores.cpu().numpy()
            pp_triu = np.triu(pp_matrix, k=1)
            pp_scores = pp_triu[pp_triu > 0]
            if len(pp_scores) > 0:
                all_distributions.append(('P-P (Protein-Protein)', pp_scores, 'green', 0.5))
        
        # Plot all distributions with density normalization
        max_score = max(scores.max() for _, scores, _, _ in all_distributions)
        bins = np.linspace(0, max_score, 50)
        
        for label, scores, color, alpha in all_distributions:
            ax.hist(scores, bins=bins, alpha=alpha, label=f'{label} (n={len(scores)})', 
                    color=color, edgecolor='black', linewidth=0.5, density=True)
            
            # Add vertical line for mean
            mean_val = scores.mean()
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.7,
                      label=f'{label} mean: {mean_val:.4f}')
        
        ax.set_xlabel('Attention Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Comparison of Attention Score Distributions Across Edge Types', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "attention_comparison_overlay.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Overlaid comparison saved to: {results_dir}/attention_comparison_overlay.png")
        
        print("    Attention map visualizations complete")

def main():
    """Main training function with support for multiple experiments"""
    
    parser = argparse.ArgumentParser(description='GNN Contrastive Model with Edge Masking')
    parser.add_argument('--num_experiments', type=int, default=1,
                       help='Number of experiments to run (default: 1)')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting seed value (default: 42)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory for single experiment results (default: results)')
    args = parser.parse_args()
    
    print("="*80)
    print("GNN CONTRASTIVE MODEL WITH EDGE MASKING")
    print("="*80)
    
    if args.num_experiments > 1:
        # Run multiple experiments
        run_multiple_experiments(
            num_experiments=args.num_experiments,
            start_seed=args.start_seed
        )
    else:
        # Run single experiment (original behavior)
        # Set seed for reproducibility
        set_seed(args.start_seed)
        
        # Create results directory
        results_dir = args.results_dir
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results will be saved to: {results_dir}/")
        
        # Load data
        print("\nLoading drug-target interaction data...")
        try:
            with open('gnn_graph_data.pkl', 'rb') as f:
                data = pickle.load(f)
            print("Data loaded successfully!")
            validation_categories, suitable_categories = analyze_meaningful_validation_categories(data)
    
        except FileNotFoundError:
            print("ERROR: gnn_graph_data.pkl not found!")
            print("Please ensure the data file is in the current directory.")
            return
        
        
        # Print data statistics
        print(f"\nData Statistics:")
        print(f"  Number of compounds: {data['num_compounds']}")
        print(f"  Number of proteins: {data['num_proteins']}")
        print(f"  Total nodes: {len(data['node_features'])}")
        print(f"  Total edges: {len(data['edge_index'][0])}")
        print(f"  Node feature dimension: {data['node_features'].shape[1]}")
        
        # Initialize model with edge masking ENABLED
        input_dim = data['node_features'].shape[1]
        hidden_dim = 128
        output_dim = 64
        num_layers = 3
        edge_dim = data.get('edge_attr', np.array([[]])).shape[1] if 'edge_attr' in data and data['edge_attr'].size > 0 else 1
        
        print(f"\nModel Configuration:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Number of GNN layers: {num_layers}")
        print(f"  Edge feature dimension: {edge_dim}")
        print(f"  Edge masking: ENABLED")
        
        model = DrugTargetGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            edge_dim=edge_dim,
            use_edge_mask=True  # MANDATORY for 3-phase training
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        # Train model using 3-phase approach
        print("\nStarting 3-phase training...")
        print("Phase 1: Standard triplet loss training")
        print("Phase 2: Learning generalizable edge masking (EXTENDED)")
        print("Phase 3: Classification head training with hard negative mining")
        
        start_time = time.time()
        
        trained_model, embeddings, test_results = train_model_3_phase(
            model,
            data,
            num_epochs_phase1=100,   # Reduced to prevent overfitting
            num_epochs_phase2=100,  # Increased for better edge masking learning
            num_epochs_phase3=30,   # Reduced to prevent overfitting
            lr=0.0001,              # Lower learning rate for better convergence
            test_ratio=0.15,
            val_ratio=0.15,
            results_dir=results_dir
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED")
        print(f"{'='*80}")
        print(f"Total training time: {training_time/60:.2f} minutes")
        print(f"Final test results:")
        print(f"  - Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"  - ROC-AUC: {test_results['test_roc_auc']:.4f}")
        print(f"  - Precision: {test_results['test_precision']:.4f}")
        print(f"  - Recall: {test_results['test_recall']:.4f}")
        print(f"  - F1-Score: {test_results['test_f1']:.4f}")
        print(f"  - MCC: {test_results['test_mcc']:.4f}")
        
        # Save model and results
        print(f"\nSaving model and results...")

        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': {
                'input_dim': trained_model.convs[0].lin.in_features,  # Get from first layer
                'hidden_dim': trained_model.convs[0].lin.out_features,  # Get from first layer output
                'output_dim': trained_model.convs[-1].lin.out_features,  # Get from last layer
                'num_layers': len(trained_model.convs),
                'edge_dim': trained_model.convs[0].edge_lin.in_features,  # Get from edge layer
                'use_edge_mask': trained_model.use_edge_mask,
            },
            'test_results': test_results,
            'embeddings': embeddings.cpu().detach().numpy() if embeddings is not None else None,
            'training_completed': True
        }, os.path.join(results_dir, 'trained_model.pth'))
        
        # Convert test results to JSON-serializable format
        test_results_json = convert_numpy_to_python(test_results)
        
        # Save test results as JSON
        with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results_json, f, indent=2)
        
        print(f"Model saved to: {results_dir}/trained_model.pth")
        print(f"Test results saved to: {results_dir}/test_results.json")
        print(f"All visualizations saved to: {results_dir}/")
        
        print(f"\n{'='*80}")
        print("Training and evaluation complete!")
        print(f"{'='*80}")
        
        return trained_model, embeddings, test_results

if __name__ == "__main__":
    main()