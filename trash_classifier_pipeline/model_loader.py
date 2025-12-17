
import numpy as np
import cv2
import pickle
from pathlib import Path
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    joblib = None


class ModelLoader:

    TARGET_SIZE = (300, 300)

    CLASS_NAMES = {
        0: "Cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "Trash"
    }

    def __init__(self, pipeline_dir="pipeline", model_type="svm"):
        self.pipeline_dir = Path(pipeline_dir)
        self.model_type = model_type

        self.classifier = None
        self.scaler = None
        self.feature_extractor = None
        self.class_to_idx = None
        self.idx_to_class = None

    def load_all(self):
        print("Loading models from", self.pipeline_dir)

        classifier_path = self.pipeline_dir / f"{self.model_type}_model.pkl"
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")

        try:
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
            try:
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                print(f"✓ Loaded {self.model_type.upper()} classifier (using pickle)")
            except Exception as e:
                raise IOError(f"Failed to load classifier from {classifier_path}. Error: {e}")

        scaler_path = self.pipeline_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        try:
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
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✓ Loaded feature scaler (using pickle)")
            except Exception as e:
                raise IOError(f"Failed to load scaler from {scaler_path}. Error: {e}")

        class_map_path = self.pipeline_dir / "class_mapping.pkl"
        if class_map_path.exists():
            try:
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
                try:
                    with open(class_map_path, 'rb') as f:
                        mappings = pickle.load(f)
                    print("✓ Loaded class mappings (using pickle)")
                except Exception as e:
                    raise IOError(f"Failed to load class mapping from {class_map_path}. Error: {e}")
            
            if isinstance(mappings, dict):
                if 'idx_to_class' in mappings:
                    self.idx_to_class = mappings['idx_to_class']
                    self.class_to_idx = mappings.get('class_to_idx', {v: k for k, v in mappings['idx_to_class'].items()})
                elif 'class_to_idx' in mappings:
                    self.class_to_idx = mappings['class_to_idx']
                    self.idx_to_class = {v: k for k, v in mappings['class_to_idx'].items()}
                else:
                    self.idx_to_class = mappings
                    self.class_to_idx = {v: k for k, v in mappings.items()}
        else:
            self.idx_to_class = self.CLASS_NAMES
            self.class_to_idx = {v: k for k, v in self.CLASS_NAMES.items()}
            print("⚠ Using default class mappings")

        print("Building EfficientNetB3 feature extractor...")
        self._build_feature_extractor()
        print("✓ Feature extractor ready")

        print("\n✅ All models loaded successfully!\n")

    def _build_feature_extractor(self):
        base = EfficientNetB3(
            weights="imagenet",
            include_top=False,
            input_shape=(self.TARGET_SIZE[0], self.TARGET_SIZE[1], 3)
        )

        x = GlobalAveragePooling2D()(base.output)
        x = Dropout(0.3)(x)

        self.feature_extractor = Model(inputs=base.input, outputs=x)

    def preprocess_image(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        image_resized = cv2.resize(image_rgb, self.TARGET_SIZE)

        image_array = np.expand_dims(image_resized, axis=0)

        image_preprocessed = preprocess_input(image_array.astype("float32"))

        return image_preprocessed

    def extract_features(self, image):
        features = self.feature_extractor.predict(image, verbose=0)
        return features

    def predict(self, image, return_probabilities=True, confidence_threshold=0.5):
        preprocessed = self.preprocess_image(image)

        features = self.extract_features(preprocessed)

        features_scaled = self.scaler.transform(features)

        prediction = self.classifier.predict(features_scaled)[0]

        confidence = 0.0
        if return_probabilities:
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba(features_scaled)[0]
                confidence = float(np.max(probs))
            else:
                confidence = 0.5

        if confidence < confidence_threshold:
            prediction = 6
            class_name = "Unknown"
        else:
            class_name = self.idx_to_class.get(prediction, "Unknown")

        if return_probabilities:
            return prediction, class_name, confidence
        else:
            return prediction, class_name

    def predict_batch(self, images, confidence_threshold=0.5):
        results = []
        for image in images:
            result = self.predict(
                image,
                return_probabilities=True,
                confidence_threshold=confidence_threshold
            )
            results.append(result)
        return results
