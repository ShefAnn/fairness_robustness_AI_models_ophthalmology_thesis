# Fairness & Robustness of AI Models for Diabetic Retinopathy Detection

... Still in process ...

This repository contains the implementation developed for my Master’s thesis focused on evaluating **fairness and robustness of deep learning models** for diabetic retinopathy (DR) classification from fundus images.

The project compares **specialized medical foundation models** and **general-purpose multimodal LLMs** to assess whether non-medical models can perform competitively in a clinical image classification setting.

---

## Project Objectives

- Compare fairness across demographic subgroups (e.g., age, sex)
- Evaluate robustness under distribution shifts and noisy inputs
- Benchmark specialized vs. non-specialized multimodal models
- Apply In-Context Learning (ICL) for multimodal LLM adaptation
- Perform statistical comparison using bootstrapping

---

## Models Evaluated

- **RETFound** (medical foundation model) - https://huggingface.co/YukunZhou/RETFound_mae_natureCFP and https://github.com/rmaphoh/RETFound. The big part of code in main_finetune.py, extract_f_p_RETFound_noise.py, extract_f_p_RETFound.py is inspired by: https://github.com/msayhan/ICL-Ophthalmology-Public
- **MedGemma 4B** (Google Health) - https://huggingface.co/google/medgemma-4b-pt and https://github.com/Google-Health/medgemma
- **Gemini 2.5 Flash** - https://gemini.google.com; Basics of using API : https://ai.google.dev/gemini-api/docs
- **DeepSeek VL 1.3B Chat** - https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat and https://github.com/deepseek-ai/DeepSeek-VL/blob/main/README.md
- **DeepSeek VL 7B Chat** - https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat and https://github.com/deepseek-ai/DeepSeek-VL/blob/main/README.md

Models were accessed via official Hugging Face repositories or APIs.  
Pretrained weights are not included in this repository.

---

## Datasets

- **BRSET (Brazilian Ophthalmological Dataset)** - https://physionet.org/content/brazilian-ophthalmological/1.0.0/
- **MESSIDOR-2** - https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess

Due to licensing restrictions:
- Medical images are not included
- Only column names are provided
- Users must obtain datasets from official sources

---

## Technical Highlights

- PyTorch-based fine-tuning pipelines
- Multimodal inference with API-based LLMs
- In-Context Learning evaluation framework
- Fairness metrics across subgroups
- Robustness testing under noise perturbations
- Bootstrap-based statistical comparison
