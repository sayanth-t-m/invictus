import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from fer import FER  # For emotion detection
import warnings
import logging

# Constants
BASELINE_DURATION = 10  # seconds
MAX_FRAMES = 10000  # Limit stored frames
FRAME_SCALE = 0.5  # Downscale frames for performance
BLINK_MIN_DURATION = 0.1  # seconds
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LieDetectorApp:
    """Main application class for the lie detection system."""
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Lie Detection System (Experimental)")
        self.root.geometry("1280x720")
        self.root.configure(bg="#f0f0f0")
        
        # Show disclaimer
        messagebox.showwarning(
            "Disclaimer",
            "This is an experimental tool for research purposes. Lie detection based on facial analysis is not definitive and may produce inaccurate results. Use with caution."
        )
        
        # Initialize detector
        self.detector = EnhancedLieDetector()
        
        # Initialize progress tracking variable
        self.progress_var = tk.DoubleVar(value=0)  # Move this line here
        
        # Setup UI
        self.setup_ui()
        
        # Variables
        self.is_recording = False
        self.video_source = 0
        self.cap = None
        self.frame_data = []
        self.baseline_captured = False
        self.analysis_in_progress = False
        self.is_capturing_baseline = False
        self.baseline_data = []
        self.recording_duration = 30
        self.start_time = 0
        self.video_thread = None
        self.running = True
        
    def setup_ui(self):
        """Configure the tkinter GUI."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10))
        style.configure('TLabel', font=('Arial', 12))
        style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        
        # Left panel
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video frame
        self.video_frame = ttk.LabelFrame(self.left_panel, text="Video Feed")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control frame
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Controls")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_frame = ttk.Frame(self.control_frame)
        self.btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_start = ttk.Button(self.btn_frame, text="Start Camera", command=self.start_camera)
        self.btn_start.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_capture_baseline = ttk.Button(self.btn_frame, text="Capture Baseline", command=self.capture_baseline, state='disabled')
        self.btn_capture_baseline.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_start_analysis = ttk.Button(self.btn_frame, text="Start Analysis", command=self.start_analysis, state='disabled')
        self.btn_start_analysis.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_stop = ttk.Button(self.btn_frame, text="Stop", command=self.stop_recording, state='disabled')
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_save_results = ttk.Button(self.btn_frame, text="Save Results", command=self.save_results)
        self.btn_save_results.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Settings
        self.settings_frame = ttk.LabelFrame(self.control_frame, text="Settings")
        self.settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.settings_frame, text="Recording Duration (seconds):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.duration_var = tk.StringVar(value="30")
        self.duration_entry = ttk.Entry(self.settings_frame, textvariable=self.duration_var, width=5)
        self.duration_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(self.settings_frame, text="Sensitivity:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        self.sensitivity_scale = ttk.Scale(self.settings_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                         variable=self.sensitivity_var, length=200)
        self.sensitivity_scale.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(self.settings_frame, text="Question:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(self.settings_frame, textvariable=self.question_var, width=50)
        self.question_entry.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky='w')
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(self.control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Right panel
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Metrics frame
        self.metrics_frame = ttk.LabelFrame(self.right_panel, text="Real-time Metrics")
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        metrics_indicators = ttk.Frame(self.metrics_frame)
        metrics_indicators.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(metrics_indicators, text="Eye Movement (EAR):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.ear_var = tk.StringVar(value="0.0")
        ttk.Label(metrics_indicators, textvariable=self.ear_var).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(metrics_indicators, text="Microexpressions:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.micro_var = tk.StringVar(value="None")
        ttk.Label(metrics_indicators, textvariable=self.micro_var).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(metrics_indicators, text="Facial Asymmetry:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.asymmetry_var = tk.StringVar(value="0.0")
        ttk.Label(metrics_indicators, textvariable=self.asymmetry_var).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(metrics_indicators, text="Dominant Emotion:").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.emotion_var = tk.StringVar(value="Neutral")
        ttk.Label(metrics_indicators, textvariable=self.emotion_var).grid(row=3, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(metrics_indicators, text="Blink Rate:").grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.blink_var = tk.StringVar(value="0 bpm")
        ttk.Label(metrics_indicators, textvariable=self.blink_var).grid(row=4, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(metrics_indicators, text="Deception Probability:").grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.deception_var = tk.StringVar(value="0%")
        ttk.Label(metrics_indicators, textvariable=self.deception_var, font=('Arial', 12, 'bold')).grid(row=5, column=1, padx=5, pady=5, sticky='w')
        
        # Graph frame
        self.graph_frame = ttk.LabelFrame(self.right_panel, text="Real-time Analysis")
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(self.right_panel, text="Analysis Results")
        self.results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.results_text = tk.Text(self.results_frame, height=6, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.insert(tk.END, "Note: Results are experimental and not definitive.\n")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def start_camera(self):
        """Start the webcam feed."""
        if self.is_recording:
            messagebox.showinfo("Info", "Camera is already running.")
            return
        
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                raise ValueError("Could not open video source")
            
            self.is_recording = True
            self.btn_start.config(state='disabled')
            self.btn_capture_baseline.config(state='normal')
            self.btn_stop.config(state='normal')
            self.status_var.set("Camera started")
            
            # Start video processing thread
            self.running = True
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()
            self.update_gui()
        except Exception as e:
            logging.error(f"Camera error: {e}")
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def process_video(self):
        """Process video frames in a separate thread."""
        frame_count = 0
        while self.is_recording and self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning("Failed to capture frame")
                    continue
                
                # Skip every other frame for performance
                frame_count += 1
                if frame_count % 2 != 0:
                    continue
                
                # Downscale frame
                frame = cv2.resize(frame, None, fx=FRAME_SCALE, fy=FRAME_SCALE)
                
                # Process frame
                processed_frame, metrics = self.detector.process_frame(frame)
                
                # Store metrics
                if self.is_capturing_baseline:
                    self.baseline_data.append(metrics)
                if self.analysis_in_progress:
                    current_time = time.time() - self.start_time
                    metrics['time'] = current_time
                    if len(self.frame_data) < MAX_FRAMES:
                        self.frame_data.append(metrics)
                    else:
                        self.frame_data.pop(0)
                        self.frame_data.append(metrics)
                    
                    # Check duration
                    if current_time >= self.recording_duration:
                        self.root.after(0, self.stop_recording)
                
                # Update display frame
                display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                display_img = Image.fromarray(display_frame)
                display_img = self.resize_image(display_img, VIDEO_WIDTH, VIDEO_HEIGHT)
                
                # Store image to avoid garbage collection
                with threading.Lock():
                    self._display_img = display_img
                    self._photo = ImageTk.PhotoImage(image=display_img)
                    self._metrics = metrics
                
            except Exception as e:
                logging.error(f"Frame processing error: {e}")
                continue
            
            time.sleep(0.01)  # Prevent thread from hogging CPU
    
    def update_gui(self):
        """Update GUI elements from the main thread."""
        if not self.is_recording:
            return
        
        try:
            with threading.Lock():
                if hasattr(self, '_photo'):
                    self.video_label.config(image=self._photo)
                    self.video_label.image = self._photo  # Prevent garbage collection
                if hasattr(self, '_metrics') and self._metrics:
                    self.update_metrics_display(self._metrics)
                    if self.analysis_in_progress:
                        self.update_realtime_graphs()
            
            # Update progress
            if self.is_capturing_baseline:
                elapsed = time.time() - self.start_time
                self.progress_var.set((elapsed / BASELINE_DURATION) * 100)
            elif self.analysis_in_progress:
                elapsed = time.time() - self.start_time
                self.progress_var.set((elapsed / self.recording_duration) * 100)
            
            self.root.after(50, self.update_gui)
        except Exception as e:
            logging.error(f"GUI update error: {e}")
    
    def resize_image(self, img, target_width, target_height):
        """Resize image while preserving aspect ratio."""
        width, height = img.size
        aspect = width / height
        if target_width / aspect <= target_height:
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            new_height = target_height
            new_width = int(target_height * aspect)
        return img.resize((new_width, new_height), Image.LANCZOS)
    
    def update_metrics_display(self, metrics):
        """Update real-time metrics in the UI."""
        self.ear_var.set(f"{metrics.get('eye_aspect_ratio', 0.0):.3f}")
        self.micro_var.set(metrics.get('microexpression', 'None'))
        self.asymmetry_var.set(f"{metrics.get('facial_asymmetry', 0.0):.3f}")
        self.emotion_var.set(metrics.get('emotion', 'Neutral'))
        self.blink_var.set(f"{metrics.get('blink_rate', 0):.1f} bpm")
        self.deception_var.set(f"{metrics.get('deception_probability', 0):.1f}%")
    
    def capture_baseline(self):
        """Capture baseline metrics for 10 seconds."""
        if not self.is_recording:
            messagebox.showinfo("Info", "Please start the camera first.")
            return
        
        self.is_capturing_baseline = True
        self.baseline_data = []
        self.start_time = time.time()
        self.status_var.set("Capturing baseline (maintain neutral expression)...")
        self.progress_var.set(0)
        self.btn_capture_baseline.config(state='disabled')
        
        self.root.after(int(BASELINE_DURATION * 1000), self.stop_baseline_capture)
    
    def stop_baseline_capture(self):
        """Finalize baseline capture."""
        self.is_capturing_baseline = False
        self.progress_var.set(100)
        
        if len(self.baseline_data) > 10:  # Require sufficient data
            try:
                self.detector.set_baseline(self.baseline_data)
                self.baseline_captured = True
                self.btn_start_analysis.config(state='normal')
                self.status_var.set("Baseline captured successfully")
                messagebox.showinfo("Success", "Baseline captured")
            except Exception as e:
                logging.error(f"Baseline error: {e}")
                messagebox.showerror("Error", f"Failed to set baseline: {str(e)}")
        else:
            self.status_var.set("Insufficient baseline data")
            messagebox.showwarning("Warning", "Not enough data for baseline")
        
        self.btn_capture_baseline.config(state='normal')
    
    def start_analysis(self):
        """Start lie detection analysis."""
        if not self.is_recording:
            messagebox.showinfo("Info", "Please start the camera first.")
            return
        if not self.baseline_captured:
            messagebox.showinfo("Info", "Please capture baseline first.")
            return
        
        try:
            duration = int(self.duration_var.get())
            if duration < 5 or duration > 300:
                raise ValueError("Duration must be between 5 and 300 seconds")
            self.recording_duration = duration
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return
        
        self.frame_data = []
        self.analysis_in_progress = True
        self.start_time = time.time()
        self.progress_var.set(0)
        
        self.detector.set_sensitivity(self.sensitivity_var.get())
        question = self.question_var.get() or "Please answer the question"
        self.status_var.set(f"Analyzing: {question}")
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Analyzing response to: {question}\n")
        self.results_text.insert(tk.END, "Note: Results are experimental and not definitive.\n")
        
        # Clear graphs
        for ax in self.ax:
            ax.clear()
        self.canvas.draw()
    
    def stop_recording(self):
        """Stop recording and analyze results."""
        self.analysis_in_progress = False
        self.is_recording = False
        self.progress_var.set(0)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.btn_start.config(state='normal')
        self.btn_capture_baseline.config(state='disabled')
        self.btn_start_analysis.config(state='disabled')
        self.btn_stop.config(state='disabled')
        self.status_var.set("Analysis stopped")
        
        if self.frame_data:
            self.analyze_results()
    
    def analyze_results(self):
        """Analyze collected data and display results."""
        if not self.frame_data:
            messagebox.showinfo("Info", "No data to analyze.")
            return
        
        self.status_var.set("Analyzing results...")
        try:
            df = pd.DataFrame(self.frame_data)
            self.plot_analysis_graphs(df)
            
            summary = self.detector.generate_analysis_summary(df)
            deception_prob = df['deception_probability'].mean()
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Question: {self.question_var.get()}\n\n")
            self.results_text.insert(tk.END, f"Average Deception Probability: {deception_prob:.1f}%\n\n")
            self.results_text.insert(tk.END, summary)
            self.results_text.insert(tk.END, "\n\nNote: Results are experimental and not definitive.")
            
            self.status_var.set("Analysis complete")
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def update_realtime_graphs(self):
        """Update graphs during analysis."""
        if not self.frame_data:
            return
        
        try:
            df = pd.DataFrame(self.frame_data[-100:])  # Last 100 frames for smoothness
            
            for ax in self.ax:
                ax.clear()
            
            if 'time' in df and 'deception_probability' in df:
                self.ax[0].plot(df['time'], df['deception_probability'], 'r-', linewidth=2)
                self.ax[0].set_title('Deception Probability')
                self.ax[0].set_xlabel('Time (s)')
                self.ax[0].set_ylabel('Probability (%)')
                self.ax[0].set_ylim(0, 100)
                self.ax[0].grid(True)
            
            metrics = ['eye_aspect_ratio', 'facial_asymmetry', 'blink_rate']
            available = [m for m in metrics if m in df]
            if 'time' in df and available:
                for metric in available:
                    data = df[metric]
                    if data.max() > data.min():
                        normalized = (data - data.min()) / (data.max() - data.min()) * 100
                    else:
                        normalized = data * 0
                    self.ax[1].plot(df['time'], normalized, label=metric)
                
                self.ax[1].set_title('Normalized Metrics')
                self.ax[1].set_xlabel('Time (s)')
                self.ax[1].set_ylabel('Normalized Value (%)')
                self.ax[1].legend()
                self.ax[1].grid(True)
            
            self.canvas.draw()
        except Exception as e:
            logging.error(f"Graph update error: {e}")
    
    def plot_analysis_graphs(self, df):
        """Plot final analysis graphs."""
        try:
            for ax in self.ax:
                ax.clear()
            
            if 'time' in df and 'deception_probability' in df:
                self.ax[0].plot(df['time'], df['deception_probability'], 'r-', linewidth=2)
                self.ax[0].set_title('Deception Probability Over Time')
                self.ax[0].set_xlabel('Time (s)')
                self.ax[0].set_ylabel('Probability (%)')
                self.ax[0].set_ylim(0, 100)
                self.ax[0].grid(True)
            
            metrics = ['eye_aspect_ratio', 'facial_asymmetry', 'blink_rate']
            available = [m for m in metrics if m in df]
            if 'time' in df and available:
                for metric in available:
                    data = df[metric]
                    if data.max() > data.min():
                        normalized = (data - data.min()) / (data.max() - data.min()) * 100
                    else:
                        normalized = data * 0
                    self.ax[1].plot(df['time'], normalized, label=metric)
                
                self.ax[1].set_title('Normalized Metrics Over Time')
                self.ax[1].set_xlabel('Time (s)')
                self.ax[1].set_ylabel('Normalized Value (%)')
                self.ax[1].legend()
                self.ax[1].grid(True)
            
            self.canvas.draw()
        except Exception as e:
            logging.error(f"Graph plotting error: {e}")
    
    def save_results(self):
        """Save analysis results to files."""
        if not self.frame_data:
            messagebox.showinfo("Info", "No data to save.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not file_path:
                return
            
            df = pd.DataFrame(self.frame_data)
            df.to_csv(file_path, index=False)
            
            report_path = file_path.replace('.csv', '_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"Question: {self.question_var.get()}\n\n")
                f.write(f"Average Deception Probability: {df['deception_probability'].mean():.1f}%\n\n")
                f.write(self.detector.generate_analysis_summary(df))
                f.write("\n\nNote: Results are experimental and not definitive.")
            
            messagebox.showinfo("Success", f"Results saved to {file_path}")
            self.frame_data = []  # Clear data
        except Exception as e:
            logging.error(f"Save error: {e}")
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def cleanup(self):
        """Clean up resources on exit."""
        self.running = False
        self.is_recording = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

class EnhancedLieDetector:
    """Class for processing frames and detecting deception."""
    def __init__(self):
        """Initialize MediaPipe and ML model."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize emotion detector
        try:
            self.emotion_detector = FER(mtcnn=True)
        except Exception as e:
            logging.error(f"Emotion detector initialization failed: {e}")
            self.emotion_detector = None
        
        # Initialize ML model
        self.model = self.create_model()
        
        # Variables
        self.baseline = None
        self.blink_threshold = 0.2
        self.blink_counter = 0
        self.last_blink_time = time.time()
        self.blink_times = []
        self.sensitivity = 0.7
        self.landmark_history = []
        self.microexpression_window = 15  # ~0.5s at 30fps
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.eye_movement_history = []
        self.eye_movement_window = 30
        
        # Emotion weights for deception
        self.emotion_weights = {
            'angry': 0.2,
            'disgust': 0.2,
            'fear': 0.4,
            'happy': -0.1,
            'sad': 0.1,
            'surprise': 0.3,
            'neutral': 0.0
        }
    
    def create_model(self):
        """Create and train a simple ML model."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Synthetic training data (replace with real dataset)
        X = np.random.rand(100, 5) * [0.5, 0.1, 30, 10, 1]  # EAR, asymmetry, blink, movement, emotion_score
        y = np.random.randint(0, 2, 100)  # 0=truth, 1=deception
        model.fit(X, y)
        
        return model
    
    def set_baseline(self, baseline_data):
        """Set baseline metrics with outlier filtering."""
        if not baseline_data:
            raise ValueError("No baseline data provided")
        
        df = pd.DataFrame(baseline_data)
        
        # Filter outliers (within 2 std devs)
        def filter_outliers(series):
            mean, std = series.mean(), series.std()
            return series[(series >= mean - 2 * std) & (series <= mean + 2 * std)]
        
        metrics = {}
        for key in ['eye_aspect_ratio', 'facial_asymmetry', 'blink_rate']:
            if key in df:
                filtered = filter_outliers(df[key])
                metrics[key] = filtered.mean() if not filtered.empty else 0.0
        
        self.baseline = {
            'eye_aspect_ratio': metrics.get('eye_aspect_ratio', 0.3),
            'facial_asymmetry': metrics.get('facial_asymmetry', 0.0),
            'blink_rate': metrics.get('blink_rate', 15.0),
            'emotion': 'neutral'
        }
        
        # Adjust blink threshold
        self.blink_threshold = self.baseline['eye_aspect_ratio'] * 0.7
        logging.info(f"Baseline set: {self.baseline}")
    
    def set_sensitivity(self, sensitivity):
        """Set detection sensitivity."""
        self.sensitivity = max(0.1, min(1.0, sensitivity))
    
    def process_frame(self, frame):
        """Process a frame to extract metrics."""
        if frame is None or frame.size == 0:
            return frame, {}
        
        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.face_mesh.process(rgb_frame)
        except Exception as e:
            logging.error(f"MediaPipe error: {e}")
            results = None
        
        metrics = {
            'eye_aspect_ratio': 0.0,
            'facial_asymmetry': 0.0,
            'blink_rate': 0.0,
            'microexpression': 'None',
            'emotion': 'Neutral',
            'deception_probability': 0.0
        }
        
        if results and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            self.mp_drawing.draw_landmarks(
                display_frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            metrics = self.extract_metrics(face_landmarks, frame, display_frame)
            metrics['deception_probability'] = self.calculate_deception_probability(metrics)
            self.draw_metrics(display_frame, metrics)
        else:
            cv2.putText(display_frame, "No face detected", (10, 30), self.font, 0.7, (0, 0, 255), 2)
        
        return display_frame, metrics
    
    def extract_metrics(self, landmarks, frame, display_frame):
        """Extract metrics from landmarks."""
        h, w, _ = frame.shape
        points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks.landmark]
        
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        left_eye = [points[i] for i in left_eye_indices]
        right_eye = [points[i] for i in right_eye_indices]
        
        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0 if left_ear > 0 and right_ear > 0 else 0.0
        
        # Draw eyes
        for eye in [left_eye, right_eye]:
            eye_np = np.array(eye, dtype=np.int32)
            cv2.polylines(display_frame, [eye_np], True, (0, 255, 0), 1)
        
        # Blink detection
        current_time = time.time()
        if ear > 0 and ear < self.blink_threshold:
            self.blink_counter += 1
            if self.blink_counter >= 2 and current_time - self.last_blink_time > BLINK_MIN_DURATION:
                self.blink_times.append(current_time)
                self.last_blink_time = current_time
                cv2.putText(display_frame, "BLINK DETECTED", (10, 30), self.font, 0.7, (0, 0, 255), 2)
        else:
            self.blink_counter = 0
        
        blink_rate = self.calculate_blink_rate()
        
        # Eye movement
        self.eye_movement_history.append((left_eye[0], right_eye[0]))
        if len(self.eye_movement_history) > self.eye_movement_window:
            self.eye_movement_history.pop(0)
        
        # Microexpression detection
        self.landmark_history.append(points)
        if len(self.landmark_history) > self.microexpression_window:
            self.landmark_history.pop(0)
        microexpression = self.detect_microexpression()
        
        # Emotion detection
        emotion, emotion_score = self.detect_emotion(frame)
        
        asymmetry = self.calculate_facial_asymmetry(points)
        
        return {
            'eye_aspect_ratio': ear,
            'facial_asymmetry': asymmetry,
            'blink_rate': blink_rate,
            'microexpression': microexpression,
            'emotion': emotion
        }
    
    def calculate_eye_aspect_ratio(self, eye):
        """Calculate EAR for an eye."""
        if len(eye) != 6:
            return 0.0
        try:
            v1 = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
            v2 = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
            h = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        except:
            return 0.0
    
    def calculate_blink_rate(self):
        """Calculate blinks per minute."""
        current_time = time.time()
        self.blink_times = [t for t in self.blink_times if current_time - t < 60]
        return len(self.blink_times)
    
    def calculate_eye_movement_score(self):
        """Calculate eye movement intensity."""
        if len(self.eye_movement_history) < 2:
            return 0.0
        
        centers = [((left[0] + right[0]) / 2, (left[1] + right[1]) / 2) 
                  for left, right in self.eye_movement_history]
        
        total_distance = 0.0
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return total_distance / (len(centers) - 1)
    
    def calculate_facial_asymmetry(self, points):
        """Calculate facial asymmetry."""
        try:
            nose = points[1]
            left_cheek = points[234]
            right_cheek = points[454]
            
            left_dist = np.linalg.norm(np.array(nose) - np.array(left_cheek))
            right_dist = np.linalg.norm(np.array(nose) - np.array(right_cheek))
            
            return abs(left_dist - right_dist) / max(left_dist, right_dist, 1e-10)
        except:
            return 0.0
    
    def detect_microexpression(self):
        """Detect rapid landmark changes."""
        if len(self.landmark_history) < self.microexpression_window:
            return "None"
        
        try:
            velocities = []
            for i in range(1, len(self.landmark_history)):
                prev, curr = self.landmark_history[i-1], self.landmark_history[i]
                total_dist = sum(np.linalg.norm(np.array(p1) - np.array(p2)) 
                               for p1, p2 in zip(prev, curr))
                velocities.append(total_dist)
            
            avg_velocity = np.mean(velocities)
            if max(velocities) > avg_velocity * 2 * self.sensitivity:
                return "Sudden Movement"
            return "None"
        except:
            return "None"
    
    def detect_emotion(self, frame):
        """Detect emotion using FER."""
        if self.emotion_detector is None:
            return "Neutral", 0.0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.emotion_detector.detect_emotions(rgb_frame)
            if result:
                emotions = result[0]['emotions']
                dominant = max(emotions, key=emotions.get)
                score = emotions[dominant]
                return dominant.capitalize(), score
            return "Neutral", 0.0
        except Exception as e:
            logging.error(f"Emotion detection error: {e}")
            return "Neutral", 0.0
    
    def calculate_deception_probability(self, metrics):
        """Calculate deception probability using ML model."""
        if not self.baseline:
            return 0.0
        
        try:
            ear = metrics.get('eye_aspect_ratio', 0.0)
            asymmetry = metrics.get('facial_asymmetry', 0.0)
            blink_rate = metrics.get('blink_rate', 0.0)
            emotion = metrics.get('emotion', 'Neutral').lower()
            
            ear_dev = abs(ear - self.baseline['eye_aspect_ratio']) / (self.baseline['eye_aspect_ratio'] + 1e-10)
            blink_dev = abs(blink_rate - self.baseline['blink_rate']) / (self.baseline['blink_rate'] + 1e-10)
            asymmetry_score = asymmetry / (self.baseline['facial_asymmetry'] + 0.1)
            movement_score = self.calculate_eye_movement_score() / 10.0
            emotion_score = self.emotion_weights.get(emotion, 0.0)
            
            features = np.array([[ear_dev, asymmetry_score, blink_dev, movement_score, emotion_score]])
            prob = self.model.predict_proba(features)[0][1] * 100 * self.sensitivity
            
            return min(max(prob, 0.0), 100.0)
        except Exception as e:
            logging.error(f"Deception calculation error: {e}")
            return 0.0
    
    def draw_metrics(self, frame, metrics):
        """Draw metrics on the frame."""
        y = 30
        for key, value in metrics.items():
            if key != 'time':
                text = f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, (int, float)) else f"{key.title()}: {value}"
                cv2.putText(frame, text, (10, y), self.font, 0.5, (255, 255, 255), 1)
                y += 20
    
    def generate_analysis_summary(self, df):
        """Generate analysis summary."""
        if df.empty:
            return "No data available."
        
        try:
            summary = (
                f"Average Deception Probability: {df['deception_probability'].mean():.1f}%\n"
                f"Max Deception Probability: {df['deception_probability'].max():.1f}%\n"
                f"Average Blink Rate: {df['blink_rate'].mean():.1f} bpm\n"
                f"Average Eye Aspect Ratio: {df['eye_aspect_ratio'].mean():.3f}\n"
                f"Average Facial Asymmetry: {df['facial_asymmetry'].mean():.3f}\n"
                f"Most Common Emotion: {df['emotion'].mode()[0] if 'emotion' in df else 'Unknown'}\n"
                f"Microexpression Events: {sum(1 for x in df['microexpression'] if x != 'None')}"
            )
            return summary
        except:
            return "Error generating summary."

if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Suppress minor warnings
    root = tk.Tk()
    app = LieDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()