
# NatureMS: De Novo Structure Elucidation Pipeline

NatureMS is an open-source, hardware-optimized pipeline for elucidating structures of organic and organometallic compounds (30–3000 Da) from mass spectrometry data, including natural products, lipids, carbohydrates, and primary/secondary metabolites. It integrates advanced peak detection (Pipeline X), machine learning (MolTransformer, Chiral), and external tools (SIRIUS 5, GFN2-xTB, CFM-ID, AutoDock Vina) to deliver SMILES and detailed JSON metadata in ≤1 hour/compound on i5 CPU or Colab.

## Features

- **Input**: MS data (.mzML, .csv, PNG/JPG)
- **Output**: SMILES + JSON (formula, stereochemistry, isomers, fragments, binding analysis)
- **Pipeline**: Preprocessing, class prediction, structure assembly, stereochemistry resolution, validation, output
- **Optimization**: Colab GPU or local CPU compatible

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sangeet01/Nature_MS.git
   cd Nature_MS
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install external tools manually:
   - SIRIUS 5 (download from SIRIUS website)
   - GFN2-xTB
   - CFM-ID
   - AutoDock Vina

## Usage

Run the pipeline with an MS file:
```bash
python naturems.py input.mzML --ion-mode M+H --runtime colab --output-dir ./results
```

### Arguments

- `input`: Path to .mzML, .csv, or PNG/JPG file
- `--ion-mode`: Ionization mode (M, M+H, M-H, M+Na, M+NH4)
- `--runtime`: Environment (local or colab)
- `--output-dir`: Directory for results (default: ./results)

## Training

- **MolTransformer**: Train with SMILES + fragment data (e.g., COCONUT)
  ```python
  # Use train_molformer in mol_transformer.py
  from mol_transformer import train_molformer
  train_molformer(model, train_loader, epochs=10, device='cpu')
  ```

- **Chiral**: Train with SMILES + R/S labels
  ```python
  # Use train_stereonet in chiral.py
  from chiral import train_stereonet
  train_stereonet(model, train_loader, epochs=10, device='cpu')
  ```

- **Random Forest**: Train with compound class data in naturems.py

## Testing

- Test with sample MS data (e.g., mzML from public repositories like MassBank)
- Check `results/output.txt` for SMILES and JSON output

## Dependencies

See `requirements.txt` for Python dependencies:
- torch>=2.0.0
- torch_geometric>=2.0.0
- opencv-python>=4.5.0
- pandas>=1.3.0
- pymzml>=2.5.0
- rdkit>=2022.03.1
- scikit-learn>=1.0.0
- scipy>=1.7.0

External tools (SIRIUS 5, GFN2-xTB, CFM-ID, AutoDock Vina) need separate installation.

## License

Apache License 2.0 - see LICENSE file.

## Citation
A paper describing NatureMS will be uploaded to arXiv. Once available, please cite:
Sharma, S, Pandey, B. (2025), "NatureMS: A Hardware Optimized, Open Source Pipeline for De Novo Structure Elucidation of Organic and Organometallic Compounds from Mass Spectrometry Data," arXiv preprint.

## Contributing

Fork the repo, submit pull requests, or report issues. Contact Sangeet S. via LinkedIn for collaboration.

## Acknowledgments
This project utilized ChatGPT for code generation and preliminary testing, Gemini for debugging, and xAI for optimizations. The core algorithm was designed entirely by me, and rigorous testing is being conducted to ensure its validity and robustness.

## Contact
Contributions to Nature_MS are welcome! Please fork the repository, make your changes, and submit a pull request. For questions or to discuss potential contributions, contact [Sangeet Sharma](https://www.linkedin.com/in/sangeet-sangiit01) on LinkedIn.

PS: Sangeet's the name, a daft undergrad splashing through chemistry and code like a toddler—my titrations are a mess, and I've used my mouth to pipette.
