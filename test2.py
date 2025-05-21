import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import numpy as np
import os
import time
from pynput.keyboard import Controller, Key

# Define same model architecture
class GestureMobileNet(nn.Module):
    def __init__(self):
        super(GestureMobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 14)

    def forward(self, x):
        return self.model(x)

class GestureKeyboardController:
    def __init__(self):
        # Keyboard controller
        self.keyboard = Controller()
        self.config_file = "key_mappings.json"
        self.last_gesture = None
        self.last_time = 0
        self.cooldown = 1.0  # 1 second cooldown to avoid multiple triggers
        
        # Set up gesture recognition
        self.setup_model()
        self.setup_camera()

    def setup_model(self):
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.mobilenet_v2(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 14)
        self.model.load_state_dict(torch.load("mobilenetv2_hg14.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Define class labels (update as needed)
        self.class_names = ['Gesture_0', 'Gesture_1', 'Gesture_10', 'Gesture_11', 
                           'Gesture_12', 'Gesture_13', 'Gesture_2', 'Gesture_3', 
                           'Gesture_4', 'Gesture_5', 'Gesture_6', 'Gesture_7', 
                           'Gesture_8', 'Gesture_9']
    
    def setup_camera(self):
        # Define ROI coordinates (x, y, width, height)
        self.roi_x, self.roi_y = 0, 100
        self.roi_width, self.roi_height = 300, 300
        
        # Define minimum hand area threshold (adjust as needed)
        self.MIN_HAND_AREA = 3000  # pixels
    
    def load_key_mappings(self):
        import json
        # Default mappings
        default_mappings = {
            "0": "a", "1": "b", "2": "c", "3": "d", "4": "e",
            "5": "f", "6": "g", "7": "h", "8": "i", "9": "j",
            "10": "k", "11": "l", "12": "m", "13": "n"
        }
        
        # Try to load from file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return default_mappings
    
    def get_gesture_number(self, label):
        # Extract number from the gesture label (e.g., 'Gesture_5' -> 5)
        try:
            return int(label.split('_')[1])
        except (IndexError, ValueError):
            return None
    
    def process_gesture(self, gesture_label):
        gesture_num = self.get_gesture_number(gesture_label)
        if gesture_num is not None:
            current_time = time.time()
            
            # Only process if it's a new gesture or sufficient time has passed
            if (gesture_label != self.last_gesture or 
                (current_time - self.last_time) > self.cooldown):
                
                # Load the key mappings
                key_mappings = self.load_key_mappings()
                
                # Get the corresponding key for this gesture
                key_str = key_mappings.get(str(gesture_num), "")
                
                if key_str:
                    print(f"Detected {gesture_label}, pressing key: {key_str}")
                    
                    # Handle the key press (single key or combination)
                    self.press_key(key_str)
                    
                    # Update last gesture info
                    self.last_gesture = gesture_label
                    self.last_time = current_time
                else:
                    print(f"No key mapping found for {gesture_label} (number {gesture_num})")
    
    def press_key(self, key_str):
        """Handle pressing keys, including combinations like 'ctrl+c'"""
        # Check if this is a key combination
        if '+' in key_str:
            keys = key_str.split('+')
            modifier_keys = []
            
            # Process all keys except the last one as modifiers
            for i in range(len(keys) - 1):
                modifier = self.map_key_string_to_key_object(keys[i].strip().lower())
                modifier_keys.append(modifier)
                self.keyboard.press(modifier)
            
            # Process the last key
            final_key = self.map_key_string_to_key_object(keys[-1].strip().lower())
            
            # Press and release the final key
            self.keyboard.press(final_key)
            self.keyboard.release(final_key)
            
            # Release all modifiers in reverse order
            for modifier in reversed(modifier_keys):
                self.keyboard.release(modifier)
        else:
            # Handle single key
            key = self.map_key_string_to_key_object(key_str)
            self.keyboard.press(key)
            self.keyboard.release(key)
    
    def map_key_string_to_key_object(self, key_str):
        """Convert string key names to pynput Key objects for special keys"""
        # Check if this is a key combination (contains "+")
        if "+" in key_str:
            return key_str  # We'll handle combinations in the press_key method
            
        # Map of special key names to Key objects
        special_keys = {
            'tab': Key.tab,
            'enter': Key.enter,
            'return': Key.enter,
            'space': Key.space,
            'esc': Key.esc,
            'escape': Key.esc,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'backspace': Key.backspace,
            'delete': Key.delete,
            'ctrl': Key.ctrl,
            'ctrl_l': Key.ctrl_l,
            'ctrl_r': Key.ctrl_r,
            'alt': Key.alt,
            'alt_l': Key.alt_l, 
            'alt_gr': Key.alt_gr,
            'shift': Key.shift,
            'shift_l': Key.shift_l,
            'shift_r': Key.shift_r,
            'cmd': Key.cmd,
            'caps_lock': Key.caps_lock,
            'page_up': Key.page_up,
            'page_down': Key.page_down,
            'home': Key.home,
            'end': Key.end,
            'insert': Key.insert,
            'f1': Key.f1,
            'f2': Key.f2,
            'f3': Key.f3,
            'f4': Key.f4,
            'f5': Key.f5,
            'f6': Key.f6,
            'f7': Key.f7,
            'f8': Key.f8,
            'f9': Key.f9,
            'f10': Key.f10,
            'f11': Key.f11,
            'f12': Key.f12,
        }
        
        # Check if it's a special key
        if key_str.lower() in special_keys:
            return special_keys[key_str.lower()]
        
        # If not a special key, return the character as is (for normal keys like 'a', 'b', etc.)
        return key_str
    
    def run(self):
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        print("Starting Gesture Recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            # Draw ROI rectangle
            cv2.rectangle(frame, (self.roi_x, self.roi_y), 
                         (self.roi_x + self.roi_width, self.roi_y + self.roi_height), 
                         (0, 255, 0), 2)
        
            # Crop ROI
            roi = frame[self.roi_y:self.roi_y + self.roi_height, 
                       self.roi_x:self.roi_x + self.roi_width]
            
            # Process the ROI
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Count non-zero pixels in the mask
            hand_area = cv2.countNonZero(mask)
            
            # Only process if sufficient hand area is detected
            if hand_area > self.MIN_HAND_AREA:
                img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
                with torch.no_grad():
                    output = self.model(input_tensor)
                    
                    # Get the prediction probabilities
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred = torch.max(probabilities, 1)
                    
                    # Only consider predictions with confidence above threshold
                    confidence_threshold = 0.9  # Adjust this value as needed (0.0-1.0)
                    
                    if confidence > confidence_threshold:
                        label = self.class_names[pred.item()]
                        
                        # Draw the prediction label with confidence
                        confidence_text = f"{label} ({confidence.item():.2f})"
                        cv2.putText(frame, confidence_text, (self.roi_x, self.roi_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        
                        # Process detected gesture
                        self.process_gesture(label)
                    else:
                        # Display low confidence warning
                        cv2.putText(frame, "No gesture detected", (self.roi_x, self.roi_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                # No hand detected
                cv2.putText(frame, "No hand detected", (self.roi_x, self.roi_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
            # Add key mappings info
            cv2.putText(frame, "Using Key Mappings from file", 
                       (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add exit instructions
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Gesture Recognition & Keyboard Control", frame)
        
            # Exit on 'q' press or window close button
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty("Gesture Recognition & Keyboard Control", cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureKeyboardController()
    controller.run()