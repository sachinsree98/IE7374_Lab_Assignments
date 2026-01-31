# Wine Classifier - ML Pipeline with CI/CD

A simple machine learning project demonstrating a complete ML workflow with automated testing and continuous integration using GitHub Actions.

## Overview

This project implements a wine quality classifier using scikit-learn's Logistic Regression model trained on the Wine dataset. The classifier predicts wine cultivar categories based on 13 chemical properties including alcohol content, acidity, and color intensity.

## Features

- **ML Model**: Logistic regression classifier with standard scaling achieving 85%+ accuracy
- **Model Persistence**: Save and load trained models using pickle (.pkl) format
- **Comprehensive Testing**: 25+ pytest test cases covering initialization, training, prediction, and persistence
- **CI/CD Pipeline**: Automated GitHub Actions workflow for training, testing, and artifact storage
- **Production-Ready Structure**: Clean separation between source code, tests, and trained models

## Quick Start
```bash