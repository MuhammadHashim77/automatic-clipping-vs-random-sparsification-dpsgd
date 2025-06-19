# Automatic Clipping vs Random Sparsification in DP-SGD

This project implements and compares two different approaches to gradient clipping in Differentially Private Stochastic Gradient Descent (DP-SGD) for medical image segmentation: Random Sparsification (RS) and Automatic Clipping (AC). The comparison is performed on a pneumonia segmentation task using chest X-ray images.

## Project Importance

Differential Privacy (DP) is crucial in medical imaging for several reasons:
1. **Patient Privacy Protection**: Medical images contain sensitive information that must be protected
2. **Data Sharing**: Enables secure sharing of medical datasets for research while maintaining privacy
3. **Regulatory Compliance**: Helps meet requirements like HIPAA and GDPR
4. **Model Robustness**: DP training can improve model generalization and robustness

The choice of gradient clipping method in DP-SGD significantly impacts both privacy guarantees and model performance. This project aims to provide empirical evidence of these trade-offs.

## Methodology

### Differential Privacy Implementation
- Implemented using Opacus library
- Privacy budget tracking with (ε, δ)-DP guarantees
- Two gradient clipping approaches:
  1. **Random Sparsification (RS)**: Randomly selects gradients to clip
  2. **Automatic Clipping (AC)**: Dynamically adjusts clipping threshold

### Model Architecture
- U-Net with ResNet34 encoder
- Binary segmentation for pneumonia detection
- Dice coefficient as the primary metric

### Training Process
- DP-SGD with both RS and AC
- Privacy parameters:
  - Noise multiplier: 1.0
  - Max gradient norm: 1.0
  - Target epsilon: 8.0
  - Target delta: 1e-5

## Results

### Accuracy Comparison
The following plot shows the validation Dice coefficient over training epochs for both methods:

![Accuracy Comparison](outputs/comparison/accuracy_comparison.png)

*Placeholder for accuracy comparison plot*

### Privacy Loss Comparison
The following plot shows the privacy loss (ε) over training epochs:

![Privacy Loss](outputs/comparison/privacy_comparison.png)

*Placeholder for privacy loss plot*

### Key Findings
1. **Accuracy-Privacy Trade-off**:
   - RS typically provides stronger privacy guarantees
   - AC often achieves better model performance
   - The trade-off varies with dataset size and model complexity

2. **Computational Efficiency**:
   - RS is generally faster due to simpler gradient processing
   - AC requires additional computation for threshold adjustment

3. **Practical Implications**:
   - RS is preferred when privacy is the primary concern
   - AC is better when model performance is critical
   - Hybrid approaches might offer optimal balance

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the comparison experiment:
```bash
python test/run_dp_comparison.py
```

3. View results in the `outputs` directory:
   - `outputs/rs_training/`: Results for Random Sparsification
   - `outputs/automatic_clipping/`: Results for Automatic Clipping
   - `outputs/comparison/`: Comparison plots and metrics

## Future Work

1. **Hybrid Approaches**: Investigate combinations of RS and AC
2. **Adaptive Parameters**: Develop methods to automatically tune privacy parameters
3. **Multi-center Studies**: Extend to federated learning scenarios
4. **Privacy Auditing**: Implement comprehensive privacy auditing tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Opacus team for the DP-SGD implementation
- PyTorch Lightning for the training framework
- The medical imaging community for datasets and insights