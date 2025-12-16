"""
Model Loader for Material Stream Identification
CRITICAL: Must match exact training configuration
"""

import numpy as np
import cv2
import pickle
from pathlib import Path
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Try to import joblib (scikit-learn models are often saved with joblib)
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    joblib = None


class ModelLoader:
    """Loads trained models and performs predictions."""

    # CRITICAL: Must match training configuration EXACTLY
    TARGET_SIZE = (300, 300)  # EfficientNetB3 was trained with 300x300

    CLASS_NAMES = {
        0: "Cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "Trash"
    }

    def __init__(self, pipeline_dir="pipeline", model_type="svm"):
        """
        Initialize model loader.

        Args:
            pipeline_dir: Directory containing saved models
            model_type: Either "svm" or "knn"
        """
        self.pipeline_dir = Path(pipeline_dir)
        self.model_type = model_type

        self.classifier = None
        self.scaler = None
        self.feature_extractor = None
        self.class_to_idx = None
        self.idx_to_class = None

    def load_all(self):
        """Load all required components."""
        print("Loading models from", self.pipeline_dir)

        # 1. Load classifier (SVM or KNN)
        classifier_path = self.pipeline_dir / f"{self.model_type}_model.pkl"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")

        try:
            # Try joblib first (common for scikit-learn models)
            if HAS_JOBLIB:
                try:
                    self.classifier = joblib.load(classifier_path)
                    print(f"✓ Loaded {self.model_type.upper()} classifier (using joblib)")
                except Exception as e:
                    print(f"⚠ Joblib failed for classifier, trying pickle: {e}")
                    raise
            else:
                raise ImportError("joblib not available")
        except:
            # Fallback to pickle
            try:
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                print(f"✓ Loaded {self.model_type.upper()} classifier (using pickle)")
            except Exception as e:
                raise IOError(f"Failed to load classifier from {classifier_path}. Error: {e}")

        # 2. Load scaler (CRITICAL - must use same scaler from training)
        scaler_path = self.pipeline_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        try:
            # Try joblib first (common for scikit-learn models)
            if HAS_JOBLIB:
                try:
                    self.scaler = joblib.load(scaler_path)
                    print("✓ Loaded feature scaler (using joblib)")
                except Exception as e:
                    print(f"⚠ Joblib failed for scaler, trying pickle: {e}")
                    raise
            else:
                raise ImportError("joblib not available")
        except:
            # Fallback to pickle
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✓ Loaded feature scaler (using pickle)")
            except Exception as e:
                raise IOError(f"Failed to load scaler from {scaler_path}. Error: {e}")

        # 3. Load class mappings
        class_map_path = self.pipeline_dir / "class_mapping.pkl"
        if class_map_path.exists():
            try:
                # Try joblib first
                if HAS_JOBLIB:
                    try:
                        mappings = joblib.load(class_map_path)
                        print("✓ Loaded class mappings (using joblib)")
                    except Exception as e:
                        print(f"⚠ Joblib failed for class mapping, trying pickle: {e}")
                        raise
                else:
                    raise ImportError("joblib not available")
            except:
                # Fallback to pickle
                try:
                    with open(class_map_path, 'rb') as f:
                        mappings = pickle.load(f)
                    print("✓ Loaded class mappings (using pickle)")
                except Exception as e:
                    raise IOError(f"Failed to load class mapping from {class_map_path}. Error: {e}")
            
            # Handle different mapping formats
            if isinstance(mappings, dict):
                if 'idx_to_class' in mappings:
                    self.idx_to_class = mappings['idx_to_class']
                    self.class_to_idx = mappings.get('class_to_idx', {v: k for k, v in mappings['idx_to_class'].items()})
                elif 'class_to_idx' in mappings:
                    self.class_to_idx = mappings['class_to_idx']
                    self.idx_to_class = {v: k for k, v in mappings['class_to_idx'].items()}
                else:
                    # Assume it's a direct mapping
                    self.idx_to_class = mappings
                    self.class_to_idx = {v: k for k, v in mappings.items()}
        else:
            # Fallback to default mapping
            self.idx_to_class = self.CLASS_NAMES
            self.class_to_idx = {v: k for k, v in self.CLASS_NAMES.items()}
            print("⚠ Using default class mappings")

        # 4. Build feature extractor (MUST MATCH TRAINING)
        print("Building EfficientNetB3 feature extractor...")
        self._build_feature_extractor()
        print("✓ Feature extractor ready")

        print("\n✅ All models loaded successfully!\n")

    def _build_feature_extractor(self):
        """
        Build the EXACT same feature extractor used during training.
        CRITICAL: This must match your training code exactly!
        """
        # Match your training code exactly:
        base = EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_shape=(self.TARGET_SIZE[0], self.TARGET_SIZE[1], 3)
        )

        # Match the architecture from training
        x = GlobalAveragePooling2D()(base.output)
        x = Dropout(0.3)(x)  # Same dropout as training

        self.feature_extractor = Model(inputs=base.input, outputs=x)

    def preprocess_image(self, image):
        """
        Preprocess image for prediction.

        Args:
            image: Input image (BGR format from OpenCV or RGB)

        Returns:
            Preprocessed image ready for feature extraction
        """
        # Convert BGR to RGB if needed (OpenCV loads as BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's likely BGR (very rough heuristic)
            # In practice, camera feed is BGR from cv2.VideoCapture
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Resize to exact training size
        image_resized = cv2.resize(image_rgb, self.TARGET_SIZE)

        # Expand dimensions for batch processing
        image_array = np.expand_dims(image_resized, axis=0)

        # Apply EfficientNet preprocessing (CRITICAL!)
        image_preprocessed = preprocess_input(image_array.astype("float32"))

        return image_preprocessed

    def extract_features(self, image):
        """
        Extract CNN features from image.

        Args:
            image: Preprocessed image

        Returns:
            Feature vector
        """
        features = self.feature_extractor.predict(image, verbose=0)
        return features

    def predict(self, image, return_probabilities=True, confidence_threshold=0.5):
        """
        Predict class for input image.

        Args:
            image: Input image (raw from camera)
            return_probabilities: Whether to return confidence scores
            confidence_threshold: Minimum confidence for valid prediction

        Returns:
            prediction: Class ID
            class_name: Class name
            confidence: Prediction confidence (if return_probabilities=True)
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)

        # Extract features
        features = self.extract_features(preprocessed)

        # Scale features (CRITICAL - must use training scaler!)
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.classifier.predict(features_scaled)[0]

        # Get confidence
        confidence = 0.0
        if return_probabilities:
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba(features_scaled)[0]
                confidence = float(np.max(probs))
            else:
                # For SVM without probability=True
                confidence = 0.5  # Default fallback

        # Apply confidence threshold
        if confidence < confidence_threshold:
            prediction = 6  # Unknown class
            class_name = "Unknown"
        else:
            class_name = self.idx_to_class.get(prediction, "Unknown")

        if return_probabilities:
            return prediction, class_name, confidence
        else:
            return prediction, class_name

    def predict_batch(self, images, confidence_threshold=0.5):
        """
        Predict classes for multiple images.

        Args:
            images: List of images
            confidence_threshold: Minimum confidence for valid prediction

        Returns:
            List of (prediction, class_name, confidence) tuples
        """
        results = []
        for image in images:
            result = self.predict(
                image,
                return_probabilities=True,
                confidence_threshold=confidence_threshold
            )
            results.append(result)
        return results