#apache licensed
#!/usr/bin/env python3
# NatureMS Core Pipeline - Hardware Optimized Structure Elucidation

import argparse
import os
import json
import torch
import subprocess
import joblib # ADDED
import numpy as np
import cv2
import pandas as pd
import pymzml
from pathlib import Path
from typing import Dict, Any, List
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier

# Local imports
from molformer import MolFormer
from chiral import StereoNet, smiles_to_graph

class NatureMS:
    """Main pipeline class handling end-to-end processing"""

    def __init__(self, config: Dict[str, Any]):
        print("Initializing NatureMS pipeline...")
        self.config = config
        self.device = self._configure_hardware()

        # Initialize core models
        print("Loading MolFormer model...")
        self.molformer = MolFormer().to(self.device)
        print("Loading StereoNet model...")
        self.stereonet = StereoNet().to(self.device)

        print("Attempting to load RandomForest models...")
        try:
            # TODO: Load trained model, e.g., joblib.load('rf_classifier.joblib')
            self.rf_classifier = None
            if self.rf_classifier is None:
                print("WARNING: rf_classifier could not be loaded. Predictive performance will be affected.")
        except Exception as e:
            print(f"ERROR: Failed to load rf_classifier: {e}")
            self.rf_classifier = None

        try:
            # TODO: Load trained model, e.g., joblib.load('rf_peak_scorer.joblib')
            self.rf_peak_scorer = None
            if self.rf_peak_scorer is None:
                print("WARNING: rf_peak_scorer could not be loaded. Peak scoring might be affected.")
        except Exception as e:
            print(f"ERROR: Failed to load rf_peak_scorer: {e}")
            self.rf_peak_scorer = None

        # self.rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10)  # Train separately
        # self.rf_peak_scorer = RandomForestClassifier(n_estimators=50, max_depth=5)  # For peak scoring

        # Adduct mass mapping (Da)
        self.adduct_masses = {
            'M': 0.0, 'M+H': 1.0078, 'M-H': -1.0078, 'M+Na': 22.9898, 'M+NH4': 18.0338
        }

        # Neutral loss mapping (Da)
        self.neutral_losses = {
            'H2O': 18.0106, 'CO2': 44.0095, 'NH3': 17.0265, 'CH2O': 30.0106
        }

        # Biogenic templates
        self.templates = {
            'D-sugars': {'C2': 'R'},
            'L-amino acids': {'C_alpha': 'S'},
            'taxane': {'C6': 'R', 'C9': 'S'}
        }

        # Ensemble weights
        self.ensemble_weights = {'rules': 0.3, 'ml': 0.4, 'sirius': 0.3}

        # SMILES vocabulary for MolFormer decoding
        self.smiles_vocab = list("CcNnoOsS(=)[]1234567890@#H+-")  # Simplified SMILES characters

    def _configure_hardware(self) -> torch.device:
        """Set hardware context based on user choice"""
        if self.config['runtime'] == 'colab':
            torch.set_float32_matmul_precision('high')
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device('cpu')

    def _optimize_hardware(self, data_size: int) -> None:
        """Optimize hardware usage based on data size"""
        if self.config['runtime'] == 'colab' and torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU memory
            if data_size > 1000:  # Threshold for large datasets
                torch.cuda.set_per_process_memory_fraction(0.8)  # Limit to 80% GPU memory

    def _load_input_data(self, input_file: Path) -> Dict[str, Any]:
        """Preprocess input data with advanced peak detection"""
        print(f"Starting input data loading and preprocessing for {input_file}...")
        ext = input_file.suffix.lower()

        if ext in ['.mzml', '.mzxml']:
            run = pymzml.run.Reader(str(input_file))
            peaks = [(spec.mz, spec.i) for spec in run if spec.ms_level == 1]
            peaks = np.array(peaks)
        elif ext == '.csv':
            df = pd.read_csv(input_file, usecols=['m/z', 'intensity'])
            peaks = df[['m/z', 'intensity']].to_numpy()
        elif ext in ['.png', '.jpg']:
            img = cv2.imread(str(input_file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            entropy = -np.sum([p * np.log2(p + 1e-10) for p in np.histogram(gray, bins=256, density=True)[0]])
            noise_factor = 1.0 if entropy < 5.0 else 2.0
            blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=noise_factor)  # 5x5 per proposal
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            profile = np.mean(cleaned, axis=0)
            profile = savgol_filter(profile, window_length=11, polyorder=2)  # Savitzky-Golay
            peak_threshold = np.percentile(profile, 90)
            peaks_idx = (profile > peak_threshold) & (np.r_[profile[1:], 0] < profile) & (np.r_[0, profile[:-1]] < profile)
            mz_range = (50, 1000)
            mz_values = np.linspace(mz_range[0], mz_range[1], len(profile))
            peaks = np.array([[mz_values[i], profile[i]] for i in range(len(profile)) if peaks_idx[i]])
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if len(peaks) == 0:
            raise ValueError("No peaks detected in input data")

        # Noise filtering
        base_intensity = peaks[:, 1].max()
        peaks = peaks[peaks[:, 1] >= 0.01 * base_intensity]

        # Deisotoping with DBSCAN
        clustering = DBSCAN(eps=0.02, min_samples=2).fit(peaks[:, 0].reshape(-1, 1))
        labels = clustering.labels_
        monoisotopic_peaks = []
        for label in set(labels) - {-1}:  # Exclude noise
            cluster = peaks[labels == label]
            monoisotopic_peaks.append(cluster[cluster[:, 0].argmin()])  # Lowest m/z
        peaks = np.array(monoisotopic_peaks) if monoisotopic_peaks else peaks

        # Memory-efficient batching
        peaks_df = pd.DataFrame(peaks, columns=['m/z', 'intensity'])
        self._optimize_hardware(len(peaks_df))

        # Random Forest peak scoring
        peak_features = np.column_stack([
            peaks[:, 0],  # m/z
            peaks[:, 1],  # intensity
            [abs(peaks[i, 0] - peaks[i-1, 0]) if i > 0 else 0 for i in range(len(peaks))]  # m/z diff
        ])
        if self.rf_peak_scorer and hasattr(self.rf_peak_scorer, 'predict_proba'):
            peak_scores = self.rf_peak_scorer.predict_proba(peak_features)[:, 1]
        else:
            if not self.rf_peak_scorer:
                print("WARNING: rf_peak_scorer not loaded. Using default peak scores (ones).")
            else:
                print("WARNING: rf_peak_scorer does not have predict_proba. Using default peak scores (ones).")
            peak_scores = np.ones(len(peaks))
        peaks_df['score'] = peak_scores

        # Top fragment selection
        top_peaks = peaks_df.sort_values('score', ascending=False).head(10).to_numpy()  # Top 10 peaks
        peaks = top_peaks[:, :2]  # Keep only m/z and intensity

        peaks[:, 1] /= peaks[:, 1].max()
        precursor_mz = peaks[peaks[:, 1].argmax(), 0]
        neutral_mass = precursor_mz - self.adduct_masses[self.config['ion_mode']]

        frag_tree = {'precursor': precursor_mz, 'fragments': []}
        for mz, intensity in peaks:
            if mz < precursor_mz - 5:
                for loss, mass in self.neutral_losses.items():
                    if abs(precursor_mz - mz - mass) < 0.02:
                        frag_tree['fragments'].append({'m/z': mz, 'intensity': intensity, 'loss': loss})
                        break
                else:
                    frag_tree['fragments'].append({'m/z': mz, 'intensity': intensity})

        print("Finished input data loading and preprocessing.")
        return {'mass': neutral_mass, 'tree': frag_tree}

    def _predict_class(self, processed_data: Dict[str, Any]) -> str:
        """Compound class prediction with Random Forest"""
        print("Starting compound class prediction...")
        features = [processed_data['mass']]
        for loss in self.neutral_losses:
            features.append(1 if any(f.get('loss') == loss for f in processed_data['tree']['fragments']) else 0)
        features.append(0.0) # TODO: Implement actual 13C/12C ratio calculation from MS data if available.

        pred_class_idx = 0 # Default to first class or handle error
        if self.rf_classifier:
            try:
                pred_class_idx = self.rf_classifier.predict([features])[0]
            except Exception as e:
                print(f"ERROR: rf_classifier prediction failed: {e}. Using default class.")
        else:
            print("WARNING: rf_classifier not loaded. Using default class 'terpenes'.")

        classes = ['terpenes', 'alkaloids', 'carbohydrates', 'polyketides', 'lipids', 'shikimates', 'peptides', 'organometallics']
        predicted_class = classes[pred_class_idx] if pred_class_idx < len(classes) else 'large_organic'
        print(f"Predicted compound class: {predicted_class}")
        return predicted_class

    def _run_sirius(self, processed_data: Dict[str, Any]) -> str:
        """Execute SIRIUS with metal support"""
        print("Attempting to run SIRIUS...")
        output_dir = Path(self.config['output_dir'])
        temp_ms_file = output_dir / 'temp_sirius_input.ms'
        sirius_output_file = output_dir / 'sirius_smiles.txt'

        try:
            with open(temp_ms_file, 'w') as f:
                f.write(f"{processed_data['tree']['precursor']}\n")
                for frag in processed_data['tree']['fragments']:
                    f.write(f"{frag['m/z']}\n")

            cmd = [
                'sirius', '-i', str(temp_ms_file), '-o', str(sirius_output_file),
                '--elements', 'Fe,Mg,Co', '--profile', 'orbitrap', '--ppm-max', '10'
            ]
            print(f"Executing SIRIUS command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to handle errors manually

            if result.returncode != 0:
                error_message = f"SIRIUS execution failed with return code {result.returncode}.\nStderr: {result.stderr.strip()}\nStdout: {result.stdout.strip()}"
                print(f"ERROR: {error_message}")
                # Consider whether to raise RuntimeError or return a default/None
                # For now, let's mimic the original behavior of raising an error for SIRIUS failure
                raise RuntimeError(error_message)

            print("SIRIUS run successful.")
            with open(sirius_output_file, 'r') as f:
                sirius_result = f.read().strip()
                print(f"SIRIUS result: {sirius_result}")
                return sirius_result

        except FileNotFoundError:
            print("ERROR: SIRIUS executable not found. Please ensure it is installed and in PATH.")
            raise
        except subprocess.CalledProcessError as e: # Should be caught by check=True, but good practice if check=False
            print(f"ERROR: SIRIUS subprocess error: {e}")
            raise RuntimeError(f"SIRIUS subprocess error: {e.stderr}") from e
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during SIRIUS execution: {e}")
            raise RuntimeError(f"Unexpected error in SIRIUS: {e}") from e
        finally:
            temp_ms_file.unlink(missing_ok=True) # Clean up temp input file

    def _mz_to_tokens(self, frag_mzs: List[float]) -> torch.Tensor:
        """Convert m/z values to token indices for MolFormer"""
        tokens = [min(int(mz / 10), 299) for mz in frag_mzs]  # Map m/z to 0-299 range
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

    def _decode_smiles(self, logits: torch.Tensor) -> str:
        """Decode MolFormer logits to SMILES string"""
        token_ids = logits.argmax(-1)[0].cpu().numpy()  # Get most likely tokens
        smiles = ''
        for token_id in token_ids:
            if token_id < len(self.smiles_vocab):
                smiles += self.smiles_vocab[token_id]
            else:
                break  # Stop at unknown token
        return smiles

    def _assemble_structures(self, processed_data: Dict[str, Any]) -> List[str]:
        """Hybrid structure assembly with RL-inspired iterative refinement"""
        print("Starting structure assembly...")
        compound_class = self._predict_class(processed_data)
        neutral_mass = processed_data['mass']

        # RL-inspired iterative refinement for rule-based assembly
        best_smiles = None
        best_reward = -float('inf')
        max_iterations = 5  # Number of refinement iterations
        base_units = int(neutral_mass / 68) or 1  # Initial guess for terpene units (C5H8 = 68 Da)

        for iteration in range(max_iterations):
            # Rule-based assembly (adjust units based on iteration)
            if compound_class == 'terpenes':
                units = base_units + (iteration - max_iterations // 2)
                if units < 1:
                    units = 1
                rules_smiles = 'CC(C)=C' * units
            elif compound_class == 'peptides':
                units = max(1, base_units // 100 + (iteration - max_iterations // 2))
                rules_smiles = 'CC(N)C(=O)' * units
            else:
                rules_smiles = None

            # Evaluate the SMILES (calculate reward)
            if rules_smiles:
                mol = Chem.MolFromSmiles(rules_smiles)
                if mol:
                    calc_mass = Descriptors.ExactMolWt(mol)
                    mass_error = abs(calc_mass - neutral_mass) / neutral_mass
                    reward = 1 - mass_error  # Higher reward for lower mass error
                    if reward > best_reward:
                        best_reward = reward
                        best_smiles = rules_smiles
                    if mass_error < 0.001:  # 0.1% error threshold
                        break
            else:
                reward = -float('inf')

        # MolFormer
        frag_mzs = [processed_data['tree']['precursor']] + [f['m/z'] for f in processed_data['tree']['fragments']]
        tokens = self._mz_to_tokens(frag_mzs)
        with torch.no_grad():
            logits = self.molformer(tokens)
            ml_smiles = self._decode_smiles(logits)

        # SIRIUS
        try:
            sirius_smiles = self._run_sirius(processed_data)
        except RuntimeError as e:
            print(f"Note: SIRIUS step failed: {e}. Proceeding without SIRIUS results.")
            sirius_smiles = None # Allow pipeline to continue if SIRIUS fails

        # Use the best SMILES from RL loop if available
        rules_smiles = best_smiles if best_smiles else None
        candidates = [s for s in [rules_smiles, ml_smiles, sirius_smiles] if s]
        if not candidates:
            raise RuntimeError("No valid structures assembled")

        # Ensemble consensus with weighted voting
        if len(candidates) > 1:
            # TODO: Confidence scores are currently fixed. Implement dynamic scoring based on evidence.
            scores = {
                'rules': 0.5 if rules_smiles else 0.0,
                'ml': 0.7 if ml_smiles else 0.0,
                'sirius': 0.9 if sirius_smiles else 0.0
            }
            weighted_scores = {k: scores[k] * self.ensemble_weights[k] for k in scores}
            best_method = max(weighted_scores, key=weighted_scores.get)
            if best_method == 'rules' and rules_smiles:
                candidates = [rules_smiles]
            elif best_method == 'ml' and ml_smiles:
                candidates = [ml_smiles]
            elif best_method == 'sirius' and sirius_smiles:
                candidates = [sirius_smiles]

        print(f"Structure assembly finished. Number of candidates: {len(candidates)}")
        return candidates

    def _resolve_stereochemistry(self, candidates: List[str]) -> List[Dict[str, Any]]:
        """Resolve stereochemistry with templates and StereoNet"""
        print("Starting stereochemistry resolution...")
        stereo_candidates = []
        for smiles_idx, smiles in enumerate(candidates):
            print(f"Processing candidate {smiles_idx + 1}/{len(candidates)} for stereochemistry: {smiles}")
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            # Biogenic templates
            stereo_smiles = smiles
            for template, configs in self.templates.items():
                if template in ['D-sugars', 'L-amino acids', 'taxane']:
                    for pos, config in configs.items():
                        stereo_smiles = stereo_smiles.replace('C', f'[C@H]' if config == 'R' else '[C@@H]', 1)

            # StereoNet
            data = smiles_to_graph(stereo_smiles, self.device)
            if not data:
                continue
            with torch.no_grad():
                stereo_pred = self.stereonet(data)
                chiral_centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
                for idx, center in enumerate(chiral_centers[:10]):
                    if stereo_pred[center, 1] > 0.5:
                        stereo_smiles = stereo_smiles.replace('C', '[C@@H]', 1)
                    else:
                        stereo_smiles = stereo_smiles.replace('C', '[C@H]', 1)

            # Energy validation (MMFF94 then GFN2-xTB)
            mol = Chem.MolFromSmiles(stereo_smiles)
            if mol:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=10)
                ff = AllChem.MMFFGetMoleculeForceField(mol)
                mmff_energy = ff.CalcEnergy() if ff else float('inf')
                if mmff_energy < 10:  # Pruning threshold
                    # TODO: Specify path to xtb executable and confirm input/output format.
                    xtb_executable_path = "xtb" # Placeholder
                    gfn_energy = mmff_energy # Default to MMFF energy if xTB fails
                    try:
                        # 1. Prepare input file for GFN2-xTB (e.g., .xyz from SMILES)
                        xyz_file = Path(self.config['output_dir']) / f"{Path(stereo_smiles).stem}_temp.xyz"
                        temp_mol = Chem.MolFromSmiles(stereo_smiles) # Ensure mol object exists

                        if not temp_mol:
                            print(f"WARNING: Could not create RDKit mol from {stereo_smiles} for GFN2-xTB input generation.")
                            stereo_candidates.append({'smiles': stereo_smiles, 'energy': mmff_energy}) # Fallback to MMFF
                            continue

                        temp_mol_with_hs = Chem.AddHs(temp_mol)
                        # Actual conversion to XYZ format is complex. RDKit itself doesn't write XYZ directly.
                        # This often requires external libraries or custom code.
                        # For now, writing a dummy XYZ file.
                        # A real implementation might use Chem.MolToMolBlock(), then OpenBabel to convert .mol to .xyz
                        try:
                            with open(xyz_file, 'w') as f_xyz:
                                # Placeholder XYZ: a single carbon atom. Replace with actual conversion.
                                f_xyz.write(f"{temp_mol_with_hs.GetNumAtoms()}\n")
                                f_xyz.write(f"{stereo_smiles}\n")
                                for atom in temp_mol_with_hs.GetAtoms():
                                    pos = temp_mol_with_hs.GetConformer().GetAtomPosition(atom.GetIdx())
                                    f_xyz.write(f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n")
                        except Exception as e_xyz:
                            print(f"ERROR: Failed to write XYZ file for {stereo_smiles}: {e_xyz}")
                            stereo_candidates.append({'smiles': stereo_smiles, 'energy': mmff_energy})
                            continue

                        print(f"Attempting GFN2-xTB for {stereo_smiles}...")
                        cmd_xtb = [xtb_executable_path, str(xyz_file), "--gfn", "2", "--sp"]
                        result_xtb = subprocess.run(cmd_xtb, capture_output=True, text=True, check=False, cwd=self.config['output_dir'])

                        if result_xtb.returncode == 0:
                            for line in result_xtb.stdout.splitlines():
                                if "TOTAL ENERGY" in line: # Example parsing
                                    gfn_energy = float(line.split()[-3]) # Convert Ha to kcal/mol if needed (1 Ha = 627.5 kcal/mol)
                                    break
                            print(f"GFN2-xTB successful for {stereo_smiles}. Energy: {gfn_energy}")
                        else:
                            print(f"WARNING: GFN2-xTB failed for {stereo_smiles}. Stderr: {result_xtb.stderr.strip()}")
                        xyz_file.unlink(missing_ok=True)

                    except FileNotFoundError:
                        print(f"ERROR: xtb executable not found at {xtb_executable_path}. Cannot run GFN2-xTB.")
                    except subprocess.CalledProcessError as e_xtb_sub: # If check=True were used
                        print(f"ERROR: GFN2-xTB subprocess error for {stereo_smiles}: {e_xtb_sub.stderr}")
                    except Exception as e_xtb:
                        print(f"ERROR: An unexpected error during GFN2-xTB calculation for {stereo_smiles}: {e_xtb}")

                    stereo_candidates.append({'smiles': stereo_smiles, 'energy': gfn_energy})

        if not stereo_candidates:
            print("WARNING: No valid stereoisomers after GFN2-xTB validation.")
            # Depending on desired behavior, could raise error or return original candidates
            # For now, let's ensure it returns a list, even if empty or based on MMFF only if xTB fails for all
            # The original code raises RuntimeError, so we'll keep that if truly nothing gets appended.
            # However, the logic above now appends even if xTB fails (using mmff_energy as fallback).
            # So, this RuntimeError might only be hit if all initial SMILES are invalid.
            if not any('energy' in sc for sc in stereo_candidates): # Check if any candidate has energy
                 raise RuntimeError("No valid stereochemistry resolved after energy calculations.")

        print(f"Stereochemistry resolution finished. {len(stereo_candidates)} candidates with energy.")
        return stereo_candidates

    def _validate_candidates(self, stereo_candidates: List[Dict[str, Any]], processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate with MS/MS, energy, and docking"""
        print("Starting candidate validation...")
        if not stereo_candidates:
            print("WARNING: No candidates to validate.")
            # This case should ideally be handled by the caller or return a specific error/None
            # For now, to prevent IndexError, let's raise an error or return a dummy.
            raise ValueError("Cannot validate an empty list of stereo candidates.")

        observed_mz = [f['m/z'] for f in processed_data['tree']['fragments']]

        best_candidate = stereo_candidates[0] # Default to first if all fail validation steps
        max_score = -1 # Placeholder for a combined score if multiple checks pass

        for idx, cand in enumerate(stereo_candidates):
            print(f"Validating candidate {idx + 1}/{len(stereo_candidates)}: {cand['smiles']}")
            mol = Chem.MolFromSmiles(cand['smiles'])
            if not mol:
                print(f"WARNING: Could not create RDKit mol for validation: {cand['smiles']}")
                continue

                continue

            # CFM-ID MS/MS prediction
            # TODO: Specify path to cfmid executable and confirm input/output format.
            # TODO: Implement actual MS/MS matching and cosine similarity calculation.
            cfmid_executable_path = "cfmid" # Placeholder
            cosine_sim = 0.0 # Default if CFM-ID fails or is skipped
            print(f"Attempting CFM-ID for {cand['smiles']}...")
            try:
                cfm_input_file = Path(self.config['output_dir']) / f"{Path(cand['smiles']).stem}_cfm_input.txt"
                with open(cfm_input_file, 'w') as f_cfm: f_cfm.write(cand['smiles'])
                cfm_output_file = Path(self.config['output_dir']) / f"{Path(cand['smiles']).stem}_cfm_output.txt"

                cmd_cfm = [cfmid_executable_path, str(cfm_input_file), str(cfm_output_file)] # Simplified hypothetical cmd
                # For a real call, you'd uncomment and refine:
                # result_cfm = subprocess.run(cmd_cfm, capture_output=True, text=True, check=True) # Using check=True
                # print(f"CFM-ID successful for {cand['smiles']}.")
                # predicted_spectrum = self._parse_cfmid_output(cfm_output_file) # Placeholder
                # cosine_sim = self._calculate_cosine_similarity(observed_mz, predicted_spectrum) # Placeholder
                cosine_sim = 0.1 # Dummy value for now since subprocess is commented
                print(f"CFM-ID (mock) cosine similarity for {cand['smiles']}: {cosine_sim}")
                # cfm_input_file.unlink(missing_ok=True)
                # cfm_output_file.unlink(missing_ok=True)
            except FileNotFoundError:
                 print(f"ERROR: CFM-ID executable not found at {cfmid_executable_path}.")
            except subprocess.CalledProcessError as e_cfm:
                 print(f"ERROR: CFM-ID execution failed for {cand['smiles']}: {e_cfm.stderr}")
            except Exception as e_cfm_general:
                print(f"ERROR: An unexpected error during CFM-ID processing for {cand['smiles']}: {e_cfm_general}")

            # AutoDock Vina
            # TODO: Specify path to vina executable, receptor PDBQT.
            # TODO: Implement ligand PDBQT generation and Vina config file preparation.
            # TODO: Implement actual parsing of Vina output for docking score.
            vina_executable_path = "vina" # Placeholder
            receptor_pdbqt_path = Path(self.config.get('receptor_path', 'receptor.pdbqt')) # Example: get from config
            docking_score = 0.0 # Default if Vina fails or is skipped
            print(f"Attempting AutoDock Vina for {cand['smiles']}...")
            try:
                ligand_pdbqt_file = Path(self.config['output_dir']) / f"{Path(cand['smiles']).stem}_ligand.pdbqt"
                # Placeholder for ligand prep:
                with open(ligand_pdbqt_file, 'w') as f_pdbqt: f_pdbqt.write("# Placeholder PDBQT for " + cand['smiles'])

                vina_config_file = Path(self.config['output_dir']) / f"{Path(cand['smiles']).stem}_vina_config.txt"
                # Placeholder for Vina config:
                with open(vina_config_file, 'w') as f_cfg:
                    f_cfg.write(f"receptor = {receptor_pdbqt_path.name}\nligand = {ligand_pdbqt_file.name}\n")
                    f_cfg.write("center_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 20\nsize_y = 20\nsize_z = 20\nout = out.pdbqt\n")

                cmd_vina = [vina_executable_path, "--config", str(vina_config_file), "--log", f"{Path(cand['smiles']).stem}_vina_log.txt"]
                # For a real call:
                # result_vina = subprocess.run(cmd_vina, capture_output=True, text=True, check=True, cwd=self.config['output_dir'])
                # print(f"AutoDock Vina successful for {cand['smiles']}.")
                # docking_score = self._parse_vina_output(Path(self.config['output_dir']) / f"{Path(cand['smiles']).stem}_vina_log.txt") # Placeholder
                docking_score = -8.1 # Dummy value for now
                print(f"AutoDock Vina (mock) score for {cand['smiles']}: {docking_score}")
                # ligand_pdbqt_file.unlink(missing_ok=True)
                # vina_config_file.unlink(missing_ok=True)
            except FileNotFoundError:
                print(f"ERROR: AutoDock Vina executable not found at {vina_executable_path} or receptor not found at {receptor_pdbqt_path}.")
            except subprocess.CalledProcessError as e_vina:
                print(f"ERROR: AutoDock Vina execution failed for {cand['smiles']}: {e_vina.stderr}")
            except Exception as e_vina_general:
                print(f"ERROR: An unexpected error during AutoDock Vina processing for {cand['smiles']}: {e_vina_general}")

            # Update candidate with scores
            cand['cosine_similarity'] = cosine_sim
            cand['docking_score'] = docking_score

            # Logic to select best candidate based on combined criteria
            # This is a simple example; a more sophisticated scoring might be needed.
            current_score = (cosine_sim * 10) - cand.get('energy', float('inf'))/10 + abs(docking_score) # Higher is better
            if current_score > max_score and cosine_sim >= 0.7 and cand.get('energy', float('inf')) <= 5.0 and docking_score <= -7.0: # Adjusted thresholds
                 max_score = current_score
                 best_candidate = cand

        print(f"Candidate validation finished. Best candidate: {best_candidate.get('smiles', 'None')} with combined score factor: {max_score:.2f}")
        return best_candidate

    def _save_results(self, final_structure: Dict[str, Any], processed_data: Dict[str, Any], stereo_candidates: List[Dict[str, Any]]):
        """Generate SMILES + JSON output per proposal"""
        print(f"Saving results for structure: {final_structure.get('smiles', 'N/A')}")
        mol = Chem.MolFromSmiles(final_structure.get('smiles', ''))
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else 'Unknown'
        centers = sum(1 for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED) if mol else 0
        result = {
            'smiles': final_structure.get('smiles', ''),
            'formula': formula,
            'neutral_mass': processed_data['mass'],
            'stereochemistry': {
                'centers': centers,
                'method': 'StereoNet + Templates',
                'confidence': final_structure.get('cosine_similarity', 0.0)  # Update after training
            },
            'isomers': [{'energy_kcal/mol': cand['energy'], 'energy_rank': i + 1}
                       for i, cand in enumerate(sorted(stereo_candidates, key=lambda x: x['energy']))][:2],
            'fragments': [{'m/z': f['m/z'],
                          'formula': '#TODO_formula',  # TODO: Determine actual fragment formulas.
                          'intensity': f['intensity'],
                          'annotation': f.get('loss', 'unknown loss'),
                          'cosine_similarity': final_structure.get('cosine_similarity', 0.0)}
                          for f in processed_data['tree']['fragments']],
            # TODO: Implement actual binding analysis data population after AutoDock Vina integration.
            'binding_analysis': {},
            'warnings': ['Untrained models used']
        }
        with open(Path(self.config['output_dir']) / 'output.txt', 'w') as f:
            f.write(f"{result['smiles']}\n{json.dumps(result, indent=2)}")

    def process(self, input_file: Path):
        """Execute full processing pipeline"""
        print(f"--- Starting NatureMS processing for {input_file.name} ---")

        raw_data = self._load_input_data(input_file)
        if not raw_data:
            print("ERROR: Failed to load or process input data. Halting pipeline.")
            return

        candidates = self._assemble_structures(raw_data)
        if not candidates:
            print("ERROR: No candidates assembled. Halting pipeline.")
            return

        stereo_candidates = self._resolve_stereochemistry(candidates)
        if not stereo_candidates:
            # This might happen if all SMILES are invalid or energy calculation fails for all
            print("WARNING: No stereoisomers resolved or validated with energy. Proceeding with non-stereospecific candidates if any.")
            # Fallback: use candidates from assembly if stereochemistry resolution fails completely
            # We need to ensure they have an 'energy' key for _validate_candidates, or _validate_candidates needs to handle its absence
            # For now, if _resolve_stereochemistry returns empty, it means a RuntimeError was hit or all failed.
            # Let's assume if it's empty, we cannot proceed to validation requiring energy.
            # The original code would raise RuntimeError in _resolve_stereochemistry if it ends up empty.
            # If it proceeds (meaning it has some candidates, possibly with MMFF fallback energy), then continue.
            if not any('energy' in c for c in stereo_candidates) and candidates: # If stereochem somehow returned empty but assembly had results
                 print("Critical error in stereochemistry resolution, cannot proceed to validation.")
                 return


        final_structure = self._validate_candidates(stereo_candidates, raw_data)
        if not final_structure:
            print("ERROR: Candidate validation failed to select a final structure. Halting pipeline.")
            return

        self._save_results(final_structure, raw_data, stereo_candidates)
        print(f"--- NatureMS processing finished for {input_file.name} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NatureMS: De Novo Structure Elucidation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help='Path to input file (mzML/CSV/PNG)')
    parser.add_argument('--ion-mode', type=str, required=True,
                        choices=['M', 'M+H', 'M-H', 'M+Na', 'M+NH4'], help='Ionization mode')
    parser.add_argument('--runtime', type=str, required=True,
                        choices=['local', 'colab'], help='Execution environment')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pipeline = NatureMS(vars(args))
    pipeline.process(Path(args.input))

#end
