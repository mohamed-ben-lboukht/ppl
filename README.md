# Keystroke Timing Analysis App

This application captures keystroke timing patterns and uses machine learning models to predict user characteristics based on typing dynamics.

## Features

- Captures keystroke timing with microsecond precision (10^-6 seconds)
- Provides three different prediction models:
  - Model 1: Basic statistics-based features
  - Model 2: Histogram-based features
  - Model 3: Advanced combined features
- Predicts age, gender, handedness, and user class

## Requirements

- Docker
- Internet connection (for initial Docker image download)

## Quick Start

1. Build the Docker image:
   ```
   docker build -t keystroke-app .
   ```

2. Run the container:
   ```
   docker run -p 5000:5000 keystroke-app
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## How to Use

1. Select a model from the dropdown menu
2. Type the requested phrase in the text area
3. Click "Analyze Keystrokes" to get predictions
4. View the results in the analysis section

## Contributing Data

You can help improve our models by contributing your own keystroke data:

1. Visit the "Contribute" page
2. Type freely in the text area (minimum 50 characters)
3. The system will analyze your keystrokes and show predictions
4. Confirm if the predictions are correct or provide correct information
5. Your data will be saved anonymously and used to improve our models

## Admin Dashboard

The admin dashboard allows you to:

1. View all collected keystroke data
2. Filter data by model type and prediction accuracy
3. See detailed statistics about the collected data
4. Download the complete dataset in JSON format

## Technical Details

- Backend: Flask web server with PyTorch models
- Frontend: HTML/JavaScript for capturing keystrokes and displaying results
- Models: Neural network architecture with shared layers and task-specific heads
- Data Storage: Local JSON files for easy management and portability

## Models

Each model uses different feature extraction techniques:

- **Basic Model**: Calculates statistical features like mean, median, standard deviation, etc.
- **Histogram Model**: Creates a histogram of keystroke timings across 20 bins
- **Advanced Model**: Combines both statistical and histogram-based features

## Development

If you wish to modify or extend this application:

1. Make changes to the code
2. Rebuild the Docker image
3. Run the new container

## Dataset

The models were trained on keystroke dynamics data where users typed free text. Each keystroke timing was recorded with microsecond precision. 