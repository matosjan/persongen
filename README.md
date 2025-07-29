# Personalized Face Generation with Diffusion Models

This repository contains the code for my Bachelor's thesis at the Faculty of Computer Science, HSE University (2025). The project investigates and implements modern methods for personalized face generation using state-of-the-art generative models.

---

## Overview

This project explores the problem of generating photorealistic, identity-preserving face images based on input data (images and text prompts). With the rise of diffusion-based models, personalized generation has become more realistic and controllable. However, maintaining identity while providing flexibility in pose, style, and context remains challenging.

In this work, we investigate whether explicitly separating input information into semantically independent components can improve both identity preservation and text-image alignment.&#x20;

---

## Conducted Work

- Reviewed recent methods for personalized face generation
- Formulated a hypothesis that separating input information into semantically distinct components — such as background, body, and face — can improve both identity preservation and prompt-image alignment
- Collected and processed a custom identity-oriented dataset of facial images and captions for model training and evaluation
- Implemented training pipelines for the state-of-the-art method [PhotoMaker](https://arxiv.org/abs/2312.04461) as a baseline
- Proposed and implemented a custom method built around the hypothesis, based on [IP-Adapter](https://arxiv.org/abs/2308.06721) and PhotoMaker
- Designed experiments aimed at testing the hypothesis by comparing different methods using CLIP-T, Face-Sim metrics, and qualitative assessments

---

## Results

As a result of the conducted experiments, we found that the proposed hypothesis provides noticeable improvements in both identity preservation and prompt-image alignment. These findings suggest that semantically separating input information is a promising approach and deserves further investigation.

Our custom method, inspired by the architectural ideas of PhotoMaker and IP-Adapter and explicitly implementing the proposed hypothesis, demonstrated consistent advantages over baseline methods in terms of identity fidelity and responsiveness to prompt control.

---

## Full Report

The full thesis document (in Russian) is available as [thesis\_report.pdf](./thesis_report.pdf).

