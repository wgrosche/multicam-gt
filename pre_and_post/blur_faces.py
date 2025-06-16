import json
import os
import cv2
import numpy as np
import dlib
from PIL import Image
import torch
from typing import List, Tuple, Dict, Any

# Option 1: Use MTCNN (PyTorch-based, lightweight)
try:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=True, device='cpu')# if torch.cuda.is_available() else 'cpu')
    face_detector_type = "mtcnn"
    print("Using MTCNN for face detection")
except ImportError:
    mtcnn = None
    face_detector_type = None

# Option 2: Use ultralytics YOLOv8 (if MTCNN not available)
if mtcnn is None:
    try:
        from ultralytics import YOLO
        # This will download the model on first use
        yolo_model = YOLO('yolov8n-face.pt')  # or 'yolov8s-face.pt' for better accuracy
        face_detector_type = "yolo"
        print("Using YOLOv8 for face detection")
    except ImportError:
        yolo_model = None

# Option 3: MediaPipe (lightweight, CPU-optimized)
if mtcnn is None and 'yolo_model' not in locals():
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        mediapipe_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        face_detector_type = "mediapipe"
        print("Using MediaPipe for face detection")
    except ImportError:
        mediapipe_detector = None
        face_detector_type = "opencv"  # fallback
        print("Falling back to OpenCV face detection")

def load_frame_data(frame_id: int, json_path: str = "/cvlabdata2/home/grosche/multicam-dev/multicam-gt/sequence_01_testing_annotations.json") -> Dict[str, Any]:
    """Load frame data from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frame_data = [frame for frame in data["frames"] if int(frame["frame_id"]) == frame_id]
        if not frame_data:
            raise ValueError(f"Frame {frame_id} not found in data")
        
        return frame_data[0]
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    except Exception as e:
        raise Exception(f"Error loading frame data: {e}")

def static_path_to_absolute(static_path: str) -> str:
    """
    Converts a Django /static/... path to an absolute path on the filesystem.
    """
    relative = static_path.lstrip("/").replace("static/", "", 1)
    project_root = os.path.abspath(os.path.join(__file__, ".."))  # or hardcode base path
    return os.path.join(project_root, "gtm_hit", "static", relative)

# Initialize dlib predictor (you'll need to download the model file)
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = None
try:
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
except:
    print("Warning: Could not load dlib landmarks predictor. Falling back to simpler blur method.")

def get_rects_in_bbox(image: np.ndarray, bbox: List[List[int]]) -> List:
    """Get face rectangles within a specific bounding box using available detector."""
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return []
    
    # Extract ROI
    roi = image[y1:y2, x1:x2]
    
    try:
        if face_detector_type == "mtcnn" and mtcnn is not None:
            return get_faces_mtcnn(roi, x1, y1)
        elif face_detector_type == "yolo" and 'yolo_model' in globals():
            return get_faces_yolo(roi, x1, y1)
        elif face_detector_type == "mediapipe" and 'mediapipe_detector' in globals():
            return get_faces_mediapipe(roi, x1, y1)
        else:
            return get_faces_opencv(roi, x1, y1)
    
    except Exception as e:
        print(f"Face detection failed: {e}")
        return []

def get_faces_mtcnn(roi: np.ndarray, offset_x: int, offset_y: int) -> List:
    """Detect faces using MTCNN."""
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    
    boxes, _ = mtcnn.detect(pil_image)
    
    if boxes is None:
        return []
    
    rects = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        # Adjust coordinates to full image space
        rects.append([
            x1 + offset_x,
            y1 + offset_y, 
            x2 + offset_x,
            y2 + offset_y
        ])
    
    return rects

def get_faces_yolo(roi: np.ndarray, offset_x: int, offset_y: int) -> List:
    """Detect faces using YOLOv8."""
    results = yolo_model(roi, verbose=False)
    
    rects = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # YOLO returns [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # Adjust coordinates to full image space
                rects.append([
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x, 
                    y2 + offset_y
                ])
    
    return rects

def get_faces_mediapipe(roi: np.ndarray, offset_x: int, offset_y: int) -> List:
    """Detect faces using MediaPipe."""
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = mediapipe_detector.process(roi_rgb)
    
    if not results.detections:
        return []
    
    rects = []
    h, w = roi.shape[:2]
    
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        # Adjust coordinates to full image space
        rects.append([
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y
        ])
    
    return rects

def get_faces_opencv(roi: np.ndarray, offset_x: int, offset_y: int) -> List:
    """Fallback: Detect faces using OpenCV Haar Cascades."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    rects = []
    for (x, y, w, h) in faces:
        # Convert to [x1, y1, x2, y2] format and adjust to full image space
        rects.append([
            x + offset_x,
            y + offset_y,
            x + w + offset_x,
            y + h + offset_y
        ])
    
    return rects

def get_landmarks(image: np.ndarray, rects: List) -> List:
    """Get facial landmarks for detected faces."""
    if predictor is None:
        return []
    
    landmarks = []
    for rect in rects:
        x1, y1, x2, y2 = map(int, map(round, rect))
        dlib_rect = dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2)
        try:
            shape = predictor(image, dlib_rect)
            landmarks.append(np.array([[p.x, p.y] for p in shape.parts()]))
        except Exception as e:
            print(f"Landmark detection failed for face: {e}")
            continue
    return landmarks

def get_faceline(landmarks: List) -> List:
    """Get face contour points from landmarks."""
    routes = []
    for p in range(len(landmarks)):
        faces = []
        
        # Jawline (right to left)
        for i in range(15, -1, -1):
            faces.append(landmarks[p][i+1])
        faces.append(landmarks[p][0])
        
        # Right eyebrow to forehead
        faces.append(landmarks[p][17])
        for i in range(17, 20):
            faces.append(landmarks[p][i])
        
        # Forehead connection
        faces.append(landmarks[p][19])
        faces.append(landmarks[p][24])
        
        # Left eyebrow
        for i in range(24, 26):
            faces.append(landmarks[p][i])
        
        # Close the contour
        faces.append(landmarks[p][26])
        faces.append(landmarks[p][16])
        
        routes.append(faces)
    
    return routes

def blur_paste(routes: List, img: np.ndarray, blur_strength: int = 51) -> np.ndarray:
    """Apply blur to face regions defined by contour routes."""
    mask = np.zeros_like(img)
    
    for landmarks in routes:
        # Create convex hull for better face coverage
        hull = cv2.convexHull(np.array(landmarks))
        cv2.fillPoly(mask, [hull], (255, 255, 255))
    
    # Ensure odd kernel size
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Create blurred version of entire image
    blurred_region = cv2.GaussianBlur(img, (blur_strength, blur_strength), 21)
    
    # Combine original and blurred using mask
    result_img = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    result_img = cv2.bitwise_or(result_img, cv2.bitwise_and(blurred_region, mask))
    
    return result_img

def apply_gaussian_blur(image: np.ndarray, bbox: List[List[int]], blur_strength: int = 15) -> np.ndarray:
    """Apply simple Gaussian blur to entire bounding box region."""
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return image
    
    # Extract region of interest
    roi = image[y1:y2, x1:x2]
    
    # Apply Gaussian blur
    if blur_strength % 2 == 0:
        blur_strength += 1  # Ensure odd kernel size
    
    blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
    
    # Replace the region in original image
    image_copy = image.copy()
    image_copy[y1:y2, x1:x2] = blurred_roi
    
    return image_copy

def apply_advanced_face_blur(image: np.ndarray, bbox: List[List[int]], blur_strength: int = 51) -> np.ndarray:
    """Apply advanced face blur using the best available face detector and landmark-based masking."""
    try:
        # Detect faces within the bounding box
        rects = get_rects_in_bbox(image, bbox)
        
        if not rects:
            # No faces detected, fall back to bbox blur
            print("No faces detected in bbox, not blurring")
            return image#apply_gaussian_blur(image, bbox, blur_strength // 3)
        
        # Get landmarks if available
        if predictor is not None:
            landmarks = get_landmarks(image, rects)
            if landmarks:
                # Use landmark-based precise blurring
                routes = get_faceline(landmarks)
                return blur_paste(routes, image, blur_strength)
        return image
        # Fall back to rectangular face blurring
        image_copy = image.copy()
        for rect in rects:
            x1, y1, x2, y2 = map(int, rect)
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                face_roi = image_copy[y1:y2, x1:x2]
                if blur_strength % 2 == 0:
                    blur_strength += 1
                blurred_face = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 0)
                image_copy[y1:y2, x1:x2] = blurred_face
        
        return image_copy
    
    except Exception as e:
        print(f"Advanced face blur failed, falling back to bbox blur: {e}")
        return apply_gaussian_blur(image, bbox, blur_strength // 3)


def blur_faces_in_frame(frame_id: int, 
                       output_dir: str = "blurred_images",
                       blur_method: str = "advanced",
                       blur_strength: int = 51,
                       json_path: str = "/cvlabdata2/home/grosche/multicam-dev/multicam-gt/sequence_01_testing_annotations.json"):
    """
    Blur faces in all camera views for a given frame.
    
    Args:
        frame_id: Frame ID to process
        output_dir: Directory to save blurred images
        blur_method: "bbox" for entire bbox blur, "advanced" for smart face detection
        blur_strength: Strength of Gaussian blur (higher = more blur)
        json_path: Path to the JSON annotation file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    frame_dict = json.load(open('/cvlabdata2/home/grosche/multicam-dev/multicam-gt/interpolate_cache.json'))
    try:
        # Load frame data
        frame_data = load_frame_data(frame_id, json_path)
        annotations = frame_data["annotations"]
        
        # Group annotations per camera (same as your original code)
        cam_views = {}
        for ann in annotations:
            for cam_id, bbox in ann["projections_2d"].items():
                if cam_id not in cam_views:
                    cam_views[cam_id] = []
                cam_views[cam_id].append((ann["track_id"], bbox))
        
        # Process each camera view
        for cam_id, bboxes in cam_views.items():
            try:
                # Load image for this frame and camera
                # You'll need to implement this based on your settings
                img_path = static_path_to_absolute(frame_dict[str(frame_id)][cam_id])
                
                # For now, assuming you have a way to get the image path
                # You'll need to replace this with your actual path logic
                
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                # Apply blurring to all bounding boxes
                for track_id, bbox in bboxes:
                    if blur_method == "advanced":
                        img = apply_advanced_face_blur(img, bbox, blur_strength)
                    else:
                        img = apply_gaussian_blur(img, bbox, blur_strength)
                
                # Save blurred image
                cam_dir = os.path.join(output_dir, cam_id)
                os.makedirs(cam_dir, exist_ok=True)
                out_path = os.path.join(cam_dir, f"{os.path.basename(img_path)}")
                cv2.imwrite(out_path, img)
                print(f"[INFO] Saved blurred image: {out_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process camera {cam_id}: {e}")
                continue
    
    except Exception as e:
        print(f"[ERROR] Failed to process frame {frame_id}: {e}")

def blur_faces_batch(frame_ids: List[int], **kwargs):
    """Blur faces for multiple frames."""
    for frame_id in frame_ids:
        print(f"Processing frame {frame_id}...")
        blur_faces_in_frame(frame_id, **kwargs)

if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Blur faces in tracked bounding boxes")
    # parser.add_argument("frame_id", type=int, help="Frame ID to process")
    # parser.add_argument("--output-dir", default="blurred_images", help="Output directory")
    # parser.add_argument("--blur-method", choices=["bbox", "advanced"], 
    #                    default="advanced", help="Blurring method")
    # parser.add_argument("--blur-strength", type=int, default=51, help="Blur strength")
    # parser.add_argument("--json-path", 
    #                    default="/cvlabdata2/home/grosche/multicam-dev/multicam-gt/sequence_01_testing_annotations.json",
    #                    help="Path to annotation JSON file")
    
    # args = parser.parse_args()

    blur_faces_batch(
        frame_ids=[i for i in range(12000)],
        output_dir="blurred_images",
        blur_method="advanced",
        blur_strength=15,
        json_path="/cvlabdata2/home/grosche/multicam-dev/multicam-gt/sequence_01_annotations.json"
    )