import cv2
import streamlink
import time
import math
import torch
from collections import defaultdict, deque
from ultralytics import YOLO

url = "https://www.youtube.com/-example-"
model = "yolo11x.pt" # powerfull extra-large model

target = [0 , 2 , 3 , 5 , 7 , 16 , 18] # person , car , motorcycle , bus , truck , dog , horse
config = 0.40 # minimum confidence to keep a detection
width = 960 # resize width for speed
traill = 40 # how many past positions to remember per object
timeout = 2.0 # seconds before a "lost" object is forgotten
mpp = 0.05 # estimate for km/h raise it if speeds look too slow, lower it if they look too fast.
smoothing = 0.85 # 0-1 , higher = smoother but slower to react , lower = jumpier but more instant
grace = 0.5 # seconds to keep drawing a box even if detection drops for a frame or two
# colors to cycle through for different tracked objects.
colours = [
    (0 , 0 , 255) ,
    (255 , 255 , 255) , 
]

def text(frame , text , x , y , color = (255 , 255 , 255)):
    cv2.putText(frame , text , (x , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , (0 , 0 , 0) , 4 , cv2.LINE_AA)
    cv2.putText(frame , text , (x , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , color , 2 , cv2.LINE_AA)

def colourf(track_id):
    return colours[track_id % len(colours)] # Pick a consistent color for a given track ID

print("Checking for GPU...")
if torch.cuda.is_available():
    device = 0
    device_name = torch.cuda.get_device_name(0)
    if torch.version.hip is not None:
        gpu_kind = "AMD GPU (ROCm)" # torch built with hip means it's an AMD card, not NVIDIA
    else:
        gpu_kind = "NVIDIA GPU (CUDA)"
    print(f"Using {gpu_kind}: {device_name}")
else:
    device = "cpu"
    device_name = "CPU"
    print("No GPU found, using CPU.")

print("Loading YOLO model...")
model = YOLO(model)
try:
    model.to(device)
except Exception:
    print("Could not use GPU, switching to CPU.")
    device = "cpu"
    device_name = "CPU"

print("Connecting to stream...")
streams = streamlink.streams(url)
if not streams:
    raise RuntimeError("No stream found.")
cap = cv2.VideoCapture(streams["best"].url)
cap.set(cv2.CAP_PROP_BUFFERSIZE , 1)

# data we track over time (per object ID)
lastposition = {} # track_id -> (x, y, time) for speed
lastbox = {} # track_id -> (x1, y1, x2, y2) last known box, so it doesnt flicker
lastclass = {} # track_id -> class name, needed to redraw the box during grace period
speedsmooth = {} # track_id -> smoothed km/h value, so it doesnt jump around
trail = {} # track_id -> deque of past (x, y) points
lastseen = {} # track_id -> time last detected (used to clean up)
fps = 0
prev_time = time.time()
start_time = time.time()

# loop - runs once per video frame
while True:
    ok , frame = cap.read()
    if not ok:
        print("Stream ended.")
        break

    # resize for processing speed   
    h , w = frame.shape[:2]
    scale = width / w
    frame = cv2.resize(frame , (width , int(h * scale)))

    # detection  
    results = model.track(
        frame ,
        device = device ,
        persist = True ,
        tracker = "bytetrack.yaml" ,
        classes = target ,
        conf = config ,
        verbose = False ,
    )
    result = results[0]
    now = time.time()
    class_counts = defaultdict(int)
    tracked_count = 0

    # if found   
    if result.boxes.id is not None:
        ids = result.boxes.id.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for track_id , box , cls in zip(ids , boxes , classes):
            x1 , y1 , x2 , y2 = map(int, box)
            class_name = model.names[cls]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # speed   
            rawspeed = 0
            if track_id in lastposition:
                prev_x , prev_y , prev_t = lastposition[track_id]
                dt = max(now - prev_t, 1e-6)
                pixels_moved = math.hypot(center_x - prev_x , center_y - prev_y)
                speed_px_per_sec = pixels_moved / dt
                rawspeed = speed_px_per_sec * mpp * 3.6

            # blend the old smoothed value with the new raw reading so it doesnt jump   
            if track_id in speedsmooth:
                speedsmooth[track_id] = speedsmooth[track_id] * smoothing + rawspeed * (1 - smoothing)
            else:
                speedsmooth[track_id] = rawspeed

            lastposition[track_id] = (center_x , center_y , now)
            lastbox[track_id] = (x1 , y1 , x2 , y2)
            lastclass[track_id] = class_name
            lastseen[track_id] = now

            # motion trail
            if track_id not in trail:
                trail[track_id] = deque(maxlen = traill)
            trail[track_id].append((center_x, center_y))

    # objects to draw this frame: seen just now , plus objects seen very recently   
    # (within "grace" seconds) that got missed this exact frame - stops the box   
    # from flickering on and off when YOLO briefly loses a detection   
    idstodraw = [tid for tid , t in lastseen.items() if now - t <= grace]
    tracked_count = len(idstodraw)

    for track_id in idstodraw:
        x1 , y1 , x2 , y2 = lastbox[track_id]
        class_name = lastclass[track_id]
        color = colourf(track_id)
        class_counts[class_name] += 1

        # draw the trail as a connected line
        points = list(trail[track_id])
        for i in range(1 , len(points)):
            cv2.line(frame , points[i - 1] , points[i] , color , 2)

        # draw the box and labels using the smoothed speed
        cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , color , 2)
        text(frame , f"{class_name} #{track_id}" , x1 , max(15 , y1 - 15) , color)
        text(frame , f"~{speedsmooth[track_id]:.0f} km/h", x1 , y2 + 20 , color)

    # forget objects   
    for track_id in list(lastseen.keys()):
        if now - lastseen[track_id] > timeout:
            del lastseen[track_id]
            lastposition.pop(track_id , None)
            lastbox.pop(track_id , None)
            lastclass.pop(track_id , None)
            speedsmooth.pop(track_id , None)
            trail.pop(track_id , None)

    current_time = time.time()
    fps = fps * 0.9 + (1 / (current_time - prev_time)) * 0.1
    prev_time = current_time
    runtime = int(current_time - start_time)

    y = 30
    for line in [
        f"Device: {device_name}" ,
        f"FPS: {fps:.1f}" ,
        f"Tracked: {tracked_count}" ,
        f"Runtime: {runtime}s" ,
    ]:
        text(frame , line , 15 , y)
        y = y + 26
    y = y + 6
    for name , count in sorted(class_counts.items()):
        text(frame , f"{name}: {count}" , 15 , y , (0 , 255 , 255))
        y = y + 22

    # result   
    cv2.imshow("Drone AI Tracker" , frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release() # cleanup
cv2.destroyAllWindows()
