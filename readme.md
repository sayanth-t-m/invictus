# Advanced Lie Detection System (Experimental)

This project is an experimental lie detection system that uses facial analysis, machine learning, and emotion detection to analyze deception probability. It is designed for research purposes and should not be used for definitive conclusions.

## Features

- **Real-time Video Analysis**: Captures and processes video frames from a webcam.
- **Facial Metrics Extraction**: Calculates metrics such as Eye Aspect Ratio (EAR), facial asymmetry, blink rate, and microexpressions.
- **Emotion Detection**: Uses the FER library to detect dominant emotions.
- **Deception Probability**: Estimates deception probability using a Random Forest Classifier.
- **Baseline Calibration**: Captures baseline metrics for comparison during analysis.
- **Graphical User Interface (GUI)**: Built with Tkinter for ease of use.
- **Real-time Graphs**: Displays metrics and deception probability over time.
- **Results Export**: Saves analysis results to CSV and text files.

## Disclaimer

This tool is experimental and intended for research purposes only. Lie detection based on facial analysis is not definitive and may produce inaccurate results. Use with caution.

## Requirements

To run this project, you need the following dependencies:

- Python 3.7 or higher
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- Tkinter (built-in with Python)
- Pillow
- Pandas
- Matplotlib
- Scikit-learn
- FER
- Threading
- Logging

Install the required Python packages using the following command:

```bash
pip install opencv-python mediapipe numpy pillow pandas matplotlib scikit-learn fer