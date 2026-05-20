---
title: "Llava论文: Visual Instruction Tuning"
updatedAt: "2025-08-24T12:33:19+08:00"
tags:
  - "Research"
draft: false
legacyPath: "/notes/Research/CausalDiscovery/Papers/Llava/"
---
# Llava论文: Visual Instruction Tuning

## Architecture

Use GPT4 to generate dataset with VQA.As GPT4 is a purely languange model,we take captions and bounding boxes as the input to GPT,and thus generate 3 types of Q-A pairs.
They're like $\textbf{X}_q \ \textbf{X}_v\text{<STOP> Assistant:} \textbf{X}_C \text{<STOP>}$ 

![1755338452180](/assets/notes/Research/CausalDiscovery/Papers/image/Llava/1755338452180.png)
