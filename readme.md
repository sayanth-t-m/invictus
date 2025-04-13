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

## Key Components

### Video Processing
- Real-time webcam feed processing
- Frame scaling for performance optimization
- Face detection and landmark tracking
- Maximum frame storage limit: 10,000 frames

### Analysis Metrics
- Eye Aspect Ratio (EAR) tracking
- Blink rate detection (minimum duration: 0.1 seconds)
- Facial asymmetry analysis
- Microexpression detection
- Emotion classification
- Baseline comparison

### User Interface
- Video feed display (640x480 resolution)
- Real-time metrics display
- Interactive controls
- Progress tracking
- Settings adjustment
- Results visualization

## Requirements

### Software Requirements
- Python 3.7 or higher
- OpenCV (`cv2`) - Video processing
- MediaPipe - Face detection
- NumPy - Numerical operations
- Tkinter (built-in with Python) - GUI framework
- Pillow - Image processing
- Pandas - Data management
- Matplotlib - Graphing and visualization
- Scikit-learn - Machine learning
- FER - Facial emotion recognition
- Threading - Asynchronous processing
- Logging - Debug and error tracking

### Hardware Requirements
- Webcam with minimum 640x480 resolution
- CPU with decent processing power
- Minimum 4GB RAM recommended
- Display resolution: 1280x720 or higher

### Installation

1. Install Python 3.7 or higher from [python.org](https://python.org)
2. Install required packages:

```bash
pip install opencv-python mediapipe numpy pillow pandas matplotlib scikit-learn fer
```

## Project Structure

```
invictus/
├── invictus.py                     # Main application file
├── shape_predictor_68_face_landmarks.dat  # Pre-trained model for facial landmarks
├── logs/                           # Application logs directory
├── results/                        # Analysis results directory
│   ├── csv/                        # CSV exports
│   └── reports/                    # Text reports
└── README.md                       # Project documentation
```

## Configuration

### Constants
- BASELINE_DURATION: 10 seconds
- MAX_FRAMES: 10,000 frames
- FRAME_SCALE: 0.5
- BLINK_MIN_DURATION: 0.1 seconds
- VIDEO_WIDTH: 640 pixels
- VIDEO_HEIGHT: 480 pixels

### Settings
- Recording duration: Configurable (default: 30 seconds)
- Sensitivity: Adjustable scale (0.1 to 1.0)
- Custom question input support

## Usage Guide

1. **Initial Setup**
   - Launch the application
   - Review and accept the disclaimer
   - Ensure proper lighting conditions

2. **Baseline Capture**
   - Click "Start Camera" to initialize video feed
   - Maintain neutral expression
   - Click "Capture Baseline" (10-second duration)
   - Wait for baseline processing completion

3. **Analysis Session**
   - Enter the question in the settings panel
   - Adjust sensitivity if needed
   - Set recording duration
   - Click "Start Analysis"
   - Monitor real-time metrics

4. **Results Review**
   - Examine graphs and metrics
   - Review deception probability
   - Save results if needed

## Error Handling

The application includes comprehensive error handling for:
- Camera initialization failures
- Face detection issues
- Processing errors
- File operations
- Invalid user inputs

## Logging

Detailed logging is implemented with different levels:
- INFO: General operation information
- WARNING: Non-critical issues
- ERROR: Critical problems
- DEBUG: Development information

## Known Limitations

1. **Technical Limitations**
   - Requires consistent lighting
   - Face must be clearly visible
   - May be affected by glasses or facial hair
   - Processing delay on slower systems

2. **Accuracy Considerations**
   - Not suitable for legal or professional use
   - Results are experimental and probabilistic
   - Requires proper baseline calibration
   - Environmental factors can affect results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
1. Check existing GitHub issues
2. Create a new issue with:
   - System details
   - Error messages
   - Steps to reproduce

## Version History

- v0.1.0 - Initial release
- v0.1.1 - Bug fixes and performance improvements

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for facial landmark detection
- [FER](https://github.com/justinshenk/fer) for emotion detection
- [Scikit-learn](https://scikit-learn.org/) for machine learning

---
**Note**: This project is for research purposes only and should not be used in real-world lie detection scenarios.