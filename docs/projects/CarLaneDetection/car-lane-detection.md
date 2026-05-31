---
title: "Car Lane Detection"
description: "从 Canny 边缘检测到 Hough 直线投票，复现车道线识别的传统计算机视觉流水线。"
date: 2025-08-01
updatedAt: 2025-08-01T04:00:00.000Z
tags:
  - "Computer Vision"
  - "Image Processing"
  - "Autonomous Driving"
featured: true
status: "Implemented"
period: "2025 Summer"
role: "Algorithm implementation and experiment notes"
techStack:
  - "Python"
  - "OpenCV"
  - "Canny"
  - "Hough Transform"
links:
  - {"label":"GitHub","href":"https://github.com/6chHenry/Car-Lane-Detection","type":"repo"}
  - {"label":"Canny Note","href":"/projects/CarLaneDetection/Canny/","type":"article"}
accent: "forest"
summary: "A compact vision project for understanding the pre-deep-learning lane detection pipeline: grayscale conversion, Gaussian smoothing, gradient filtering, edge thinning, thresholding, and Hough-space voting."
---

# Car Lane Detection

This project rebuilds a classic lane detection pipeline with explicit image-processing steps. It is useful as a low-level computer vision exercise because each stage exposes an interpretable assumption: what counts as an edge, how noise is suppressed, and how line candidates accumulate votes.

The implementation notes are split into focused articles on Canny edge detection and Hough line transform.
