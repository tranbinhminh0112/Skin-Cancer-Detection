# Important Note
Certain file's outputs can only be seen once downloaded and viewed on the IDE: <br>
- **Dataset:** https://challenge.isic-archive.com/data/#2020 <br>
- **Models are saved here:**  https://drive.google.com/drive/folders/1qnQ7R894FaRHaEizO35ATVrz2C_pzOXW?usp=drive_link
- **Proposal & Final reportModels are saved here:** 0. Proposal & Final Report

# EfficientNet B0 Model Training Experiments for Skin Lesion Classification

This repository contains comprehensive experiments with EfficientNet B0 model for skin lesion classification, focusing on training time performance, evaluation metrics, skin tone generalizations, and data augmentation techniques.

##  Project Overview

This project explores various aspects of training EfficientNet B0 models for binary classification of skin lesions (benign vs malignant) using the ISIC 2020 dataset. The experiments investigate:

1. **Training Time Performance** - Comparing training efficiency across different image sizes and datasets
2. **Evaluation Metrics** - Comprehensive performance analysis using accuracy, AUC, recall, and other metrics
3. **Skin Tone Generalizations** - Analyzing model performance across different skin tones
4. **Data Augmentation Performance** - Testing the impact of augmented images on model performance
5. **Class Imbalance Handling** - Experiments with different class distributions

##  Repository Structure

### Core Training Experiments

#### 1. **33k_EfficientNet_Original_Sized.ipynb**
- **Purpose**: Baseline training with original image sizes from ISIC 2020 dataset
- **Dataset**: 33,126 images with original dimensions
- **Key Features**:
  - Uses full ISIC 2020 training dataset
  - Original image sizes (no resizing)
  - Standard train/validation/test split (80/10/10)
  - Batch size: 2, Image size: 224x224
  - 1 epoch training for baseline performance

#### 2. **33k_EfficientNet_Resized_Images.ipynb**
- **Purpose**: Training with pre-resized images for performance comparison
- **Dataset**: 33,126 images resized to 224x224
- **Key Features**:
  - Pre-processed resized images
  - Same architecture and hyperparameters as original
  - Performance comparison with original sized images
  - Training time optimization analysis

### Class Imbalance Experiments

#### 3. **EpochTraining_950_50_EfficientNet.ipynb**
- **Purpose**: Training with severe class imbalance (950 benign, 50 malignant)
- **Dataset**: 1,000 images (950 benign, 50 malignant)
- **Key Features**:
  - 10 epochs training
  - Severe class imbalance scenario
  - Model checkpointing at each epoch
  - Performance tracking across epochs

#### 4. **EpochTraining_950_50_EfficientNet_ResizedImages.ipynb**
- **Purpose**: Same class imbalance experiment with resized images
- **Dataset**: 1,000 resized images (950 benign, 50 malignant)
- **Key Features**:
  - Pre-resized images for faster training
  - Same class distribution as above
  - Training time comparison with original sizes

#### 5. **ClassImbalanceTest_AugmentedSkin_classification_950_50_epochsTest.ipynb**
- **Purpose**: Testing data augmentation impact on class imbalance
- **Dataset**: 1,000 augmented images (950 benign, 50 malignant)
- **Key Features**:
  - 15 epochs training
  - Data augmentation techniques applied
  - Performance comparison with non-augmented data
  - Enhanced model generalization

#### 6. **ClassImbalanceTest_Skin_classification_600_400_epochsTest.ipynb**
- **Purpose**: Different class balance scenario (600 benign, 400 malignant)
- **Dataset**: 1,000 images (600 benign, 400 malignant)
- **Key Features**:
  - More balanced class distribution
  - 15 epochs training
  - Performance comparison with 950:50 ratio

#### 7. **ClassImbalanceTest_AugmentedSkin_classification_600_400_epochsTest.ipynb**
- **Purpose**: Augmented data with balanced class distribution
- **Dataset**: 1,000 augmented images (600 benign, 400 malignant)
- **Key Features**:
  - Data augmentation with balanced classes
  - 15 epochs training
  - Comprehensive performance analysis

### Skin Tone Generalization Experiments

#### 8. **SFSkintone_generalization_whiteonly.ipynb**
- **Purpose**: Model performance analysis across different skin tones
- **Dataset**: 1,000 images with skin tone annotations
- **Key Features**:
  - Skin tone distribution analysis
  - Performance comparison across skin tones (Light, Medium, Dark)
  - Model fairness evaluation
  - Detailed skin tone-specific metrics

#### 9. **SFSkintone_generalization_combinedskintone_2k.ipynb**
- **Purpose**: Extended skin tone analysis with larger dataset
- **Dataset**: 2,000 images with combined skin tone data
- **Key Features**:
  - Larger dataset for more robust analysis
  - Enhanced skin tone generalization testing
  - Comprehensive fairness metrics
  - Cross-skin-tone performance evaluation

## Technical Specifications

### Model Architecture
- **Base Model**: EfficientNet B0 (pre-trained on ImageNet)
- **Classification Head**: Single output neuron for binary classification
- **Loss Function**: Binary Cross Entropy with Logits Loss
- **Optimizer**: Adam with learning rate 1e-4

### Data Processing
- **Image Size**: 224x224 pixels
- **Normalization**: ImageNet mean/std values
- **Data Augmentation**: Various techniques including rotation, flipping, color jittering
- **Train/Val/Test Split**: 80/10/10 (stratified)

### Training Configuration
- **Batch Size**: 2 (optimized for memory constraints)
- **Device**: CUDA GPU when available, CPU fallback
- **Epochs**: 1-15 depending on experiment
- **Evaluation Metrics**: Accuracy, AUC, Recall, Precision, F1-Score

## Key Findings

### Training Performance
- **Original vs Resized**: Resized images show faster training times with comparable accuracy
- **Class Imbalance**: Severe imbalance (950:50) (600-400) requires careful handling and data augmentation
- **Epoch Analysis**: Model performance stabilizes around 10-15 epochs for most experiments

### Model Performance
- **Baseline Performance**: Good performance on balanced datasets
- **Class Imbalance Impact**: Significant performance degradation on minority class
- **Data Augmentation Benefits**: Improved generalization and reduced overfitting
- **Skin Tone Generalization**: Varying performance across different skin tones

### Evaluation Metrics
- **Accuracy**: Generally high (>80%) on balanced datasets
- **AUC**: Good discriminative ability (>0.85) in most cases
- **Recall**: Varies significantly with class imbalance
- **Precision**: Affected by false positive rates

## Usage Instructions

### Prerequisites
```bash
pip install pandas numpy pillow torch torchvision scikit-learn efficientnet_pytorch
```

### Running Experiments
1. **Setup**: Mount Google Drive and install dependencies
2. **Data Preparation**: Ensure ISIC 2020 dataset is available
3. **Configuration**: Adjust paths and hyperparameters as needed
4. **Training**: Run the desired notebook for specific experiment
5. **Evaluation**: Analyze results and compare across experiments

### Data Requirements
- ISIC 2020 Training Dataset (33,126 images)
- Ground truth CSV file with labels
- Additional datasets for specific experiments (class imbalance, skin tone analysis)


## Research Contributions

1. **Training Efficiency Analysis**: Comprehensive comparison of training times across different image processing approaches
2. **Class Imbalance Solutions**: Systematic evaluation of data augmentation and sampling strategies
3. **Skin Tone Fairness**: Detailed analysis of model performance across different skin tones
4. **Data Augmentation Impact**: Quantified benefits of various augmentation techniques
5. **Epoch-wise Performance**: Detailed tracking of model performance across training epochs

## Notes

- All experiments use the same EfficientNet B0 architecture for consistency
- Training times vary significantly based on hardware and image processing
- Class imbalance experiments highlight the importance of balanced datasets
- Skin tone analysis reveals potential biases in dermatological AI models
- Data augmentation consistently improves model generalization
