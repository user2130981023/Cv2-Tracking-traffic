import cv2
import numpy as np
import time

video_path = 'video.mp4'  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

backSub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=36, detectShadows=False)

min_contour_area = 400  
min_box_area_to_render = 350  
max_object_age = 2.0  
movement_threshold = 10  

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

class TrackedObject:
    def __init__(self, contour, img_h, img_w):
        self.contour = contour
        x, y, w, h = cv2.boundingRect(contour)
        self.box = (x, y, w, h)
        self.positions = [(x, y, w, h)]
        self.last_seen = time.time()
        self.tracked_for = 1
        self.is_moving = False
        self.movement_direction = None
        self.img_height = img_h
        self.img_width = img_w

        self.object_type = self._classify_object_type()

    def _classify_object_type(self):
        """Improved object classification based on size, aspect ratio, and position."""
        x, y, w, h = self.box
        aspect_ratio = float(w) / h if h > 0 else 0
        area = w * h
        area_ratio = area / (self.img_width * self.img_height)  
        vertical_position = y / self.img_height if self.img_height > 0 else 0

        if area_ratio > 0.03 and aspect_ratio > 0.8:

            if area_ratio > 0.08:
                return "large vehicle"
            return "car"

        elif area_ratio > 0.015 and aspect_ratio > 0.7:
            return "car"

        elif aspect_ratio < 0.5 and area_ratio < 0.02 and vertical_position > 0.3:
            return "person"

        elif 0.5 <= aspect_ratio <= 0.9 and area_ratio < 0.025:
            return "bike"

        else:
            if area_ratio > 0.03:
                return "car"  
            else:
                return "unknown"

    def update(self, contour):
        """Update the tracked object with a new contour."""
        self.contour = contour
        x, y, w, h = cv2.boundingRect(contour)

        self.box = (x, y, w, h)
        self.last_seen = time.time()
        self.tracked_for += 1

        if self.tracked_for % 10 == 0:  
            self.object_type = self._classify_object_type()

        self.positions.append((x, y, w, h))
        if len(self.positions) > 15:
            self.positions.pop(0)

        if len(self.positions) >= 5:
            first_pos = self.positions[0]
            current_pos = self.positions[-1]

            first_center = (first_pos[0] + first_pos[2]//2, first_pos[1] + first_pos[3]//2)
            current_center = (current_pos[0] + current_pos[2]//2, current_pos[1] + current_pos[3]//2)

            distance = np.sqrt((current_center[0] - first_center[0])**2 + 
                              (current_center[1] - first_center[1])**2)

            dx = current_center[0] - first_center[0]
            dy = current_center[1] - first_center[1]

            if abs(dx) > abs(dy):  
                self.movement_direction = "right" if dx > 0 else "left"
            else:  
                self.movement_direction = "down" if dy > 0 else "up"

            self.is_moving = distance > movement_threshold

    def should_render(self):
        """Determine if this object should have a box rendered around it."""

        _, _, w, h = self.box
        area = w * h

        return (self.is_moving and 
                area > min_box_area_to_render and 
                self.tracked_for > 5)  

    def merge_with(self, other_obj):
        """Merge this object with another overlapping object."""

        x1, y1, w1, h1 = self.box
        x2, y2, w2, h2 = other_obj.box

        new_x = min(x1, x2)
        new_y = min(y1, y2)
        new_w = max(x1 + w1, x2 + w2) - new_x
        new_h = max(y1 + h1, y2 + h2) - new_y

        self.box = (new_x, new_y, new_w, new_h)

        self.is_moving = self.is_moving or other_obj.is_moving

        if (w2 * h2) > (w1 * h1):
            self.object_type = other_obj.object_type

        if other_obj.tracked_for > self.tracked_for:
            self.tracked_for = other_obj.tracked_for

        self.last_seen = max(self.last_seen, other_obj.last_seen)

        self.object_type = self._classify_object_type()

tracked_objects = {}
next_id = 0

print("Starting video processing. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break

    output_frame = frame.copy()
    height, width = frame.shape[:2]

    fg_mask = backSub.apply(frame)

    kernel = np.ones((5,5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            current_contours.append(contour)

    matched_ids = set()

    for contour in current_contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w//2, y + h//2

        best_match_id = None
        best_match_distance = float('inf')

        for obj_id, tracked_obj in tracked_objects.items():
            if obj_id in matched_ids:  
                continue

            tracked_x, tracked_y, tracked_w, tracked_h = tracked_obj.box
            tracked_center_x = tracked_x + tracked_w//2
            tracked_center_y = tracked_y + tracked_h//2

            distance = np.sqrt((center_x - tracked_center_x)**2 + 
                              (center_y - tracked_center_y)**2)

            iou = calculate_iou((x, y, w, h), (tracked_x, tracked_y, tracked_w, tracked_h))

            if (distance < 100 or iou > 0.2) and distance < best_match_distance:
                best_match_distance = distance
                best_match_id = obj_id

        if best_match_id is not None:
            tracked_objects[best_match_id].update(contour)
            matched_ids.add(best_match_id)
        else:

            tracked_objects[next_id] = TrackedObject(contour, height, width)
            next_id += 1

    merged_ids = set()
    for id1, obj1 in list(tracked_objects.items()):
        if id1 in merged_ids:
            continue

        for id2, obj2 in list(tracked_objects.items()):
            if id1 == id2 or id2 in merged_ids:
                continue

            iou = calculate_iou(obj1.box, obj2.box)

            if iou > 0.3:  
                obj1.merge_with(obj2)
                merged_ids.add(id2)

    for obj_id in merged_ids:
        if obj_id in tracked_objects:
            del tracked_objects[obj_id]

    current_time = time.time()
    for obj_id, tracked_obj in list(tracked_objects.items()):

        if current_time - tracked_obj.last_seen > max_object_age:
            del tracked_objects[obj_id]
            continue

        if tracked_obj.should_render():
            x, y, w, h = tracked_obj.box

            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = tracked_obj.object_type
            cv2.putText(output_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Traffic Camera Motion Tracking', output_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Video processing complete.")