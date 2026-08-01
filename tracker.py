import cv2
import streamlink
import time
import math
import torch
import queue
import threading
from collections import defaultdict , deque
from ultralytics import YOLO

url = "https://www.youtube.com/-example-"
modelname = "yolo11x.pt" # powerfull extra-large model

target = [0 , 2 , 3 , 5 , 7 , 16 , 18]
config = 0.40
width = 960
traill = 40
timeout = 2.0
mpp = 0.05
smoothing = 0.85
grace = 0.5

colours = [
    (0 , 0 , 255) ,
    (255 , 255 , 255) , 
]

class queuestream:
    def __init__(self , cap , maxbuffer = 60):
        self.cap = cap
        self.q = queue.Queue(maxsize = maxbuffer)
        self.stopped = False
        self.thread = threading.Thread(target = self.reader , daemon = True)
        self.thread.start()

    def reader(self):
        while not self.stopped:
            ok , frame = self.cap.read()
            if not ok:
                self.stopped = True
                break
            self.q.put((ok , frame))

    def read(self):
        if self.stopped and self.q.empty():
            return False , None
        try:
            return self.q.get(timeout = 2.0)
        except queue.Empty:
            return False , None

    def stop(self):
        self.stopped = True
        self.cap.release()

def text(frame , text , x , y , color = (255 , 255 , 255)):
    cv2.putText(frame , text , (x , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , (0 , 0 , 0) , 4 , cv2.LINE_AA)
    cv2.putText(frame , text , (x , y) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , color , 2 , cv2.LINE_AA)

def colourf(trackid):
    return colours[trackid % len(colours)]

print("Checking for GPU...")
if torch.cuda.is_available():
    device = 0
    devicename = torch.cuda.get_device_name(0)
    if torch.version.hip is not None:
        gpukind = "AMD GPU (ROCm)"
    else:
        gpukind = "NVIDIA GPU (CUDA)"
    print(f"Using {gpukind}: {devicename}")
else:
    device = "cpu"
    devicename = "CPU"
    print("No GPU found, using CPU.")

print("Loading YOLO model...")
model = YOLO(modelname)
try:
    model.to(device)
except Exception:
    print("Could not use GPU, switching to CPU.")
    device = "cpu"
    devicename = "CPU"

print("Connecting to stream...")
streams = streamlink.streams(url)
if not streams:
    raise RuntimeError("No stream found.")

cap = cv2.VideoCapture(streams["best"].url)
cap.set(cv2.CAP_PROP_BUFFERSIZE , 1)

streamfps = cap.get(cv2.CAP_PROP_FPS)
if streamfps <= 0 or streamfps > 120 or math.isnan(streamfps):
    streamfps = 30.0
frameduration = 1.0 / streamfps
print(f"Tracking locked to stream rate: {streamfps:.1f} FPS ({frameduration * 1000:.1f}ms per frame)")

stream = queuestream(cap , maxbuffer = 60)

print("Pre-buffering stream...")
while stream.q.qsize() < 15 and not stream.stopped:
    time.sleep(0.05)
print("Buffer filled! Launching tracker.")

lastposition = {}
lastbox = {}
lastclass = {}
speedsmooth = {}
trail = {}
lastseen = {}
fps = 0
prevtime = time.time()
starttime = time.time()

while True:
    loopstart = time.time()

    ok , frame = stream.read()
    if not ok or frame is None:
        print("Stream ended.")
        break

    h , w = frame.shape[:2]
    scale = width / w
    frame = cv2.resize(frame , (width , int(h * scale)))

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
    classcounts = defaultdict(int)
    trackedcount = 0

    if result.boxes.id is not None:
        ids = result.boxes.id.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for trackid , box , cls in zip(ids , boxes , classes):
            x1 , y1 , x2 , y2 = map(int , box)
            classname = model.names[cls]
            centerx = (x1 + x2) // 2
            centery = (y1 + y2) // 2

            rawspeed = 0
            if trackid in lastposition:
                prevx , prevy , prevt = lastposition[trackid]
                dt = max(now - prevt , 1e-6)
                pixelsmoved = math.hypot(centerx - prevx , centery - prevy)
                speedpxpersec = pixelsmoved / dt
                rawspeed = speedpxpersec * mpp * 3.6

            if trackid in speedsmooth:
                speedsmooth[trackid] = speedsmooth[trackid] * smoothing + rawspeed * (1 - smoothing)
            else:
                speedsmooth[trackid] = rawspeed

            lastposition[trackid] = (centerx , centery , now)
            lastbox[trackid] = (x1 , y1 , x2 , y2)
            lastclass[trackid] = classname
            lastseen[trackid] = now

            if trackid not in trail:
                trail[trackid] = deque(maxlen = traill)
            trail[trackid].append((centerx , centery))

    idstodraw = [tid for tid , t in lastseen.items() if now - t <= grace]
    trackedcount = len(idstodraw)

    for trackid in idstodraw:
        x1 , y1 , x2 , y2 = lastbox[trackid]
        classname = lastclass[trackid]
        color = colourf(trackid)
        classcounts[classname] += 1

        points = list(trail[trackid])
        for i in range(1 , len(points)):
            cv2.line(frame , points[i - 1] , points[i] , color , 2)

        cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , color , 2)
        text(frame , f"{classname} #{trackid}" , x1 , max(15 , y1 - 15) , color)
        text(frame , f"~{speedsmooth[trackid]:.0f} km/h" , x1 , y2 + 20 , color)

    for trackid in list(lastseen.keys()):
        if now - lastseen[trackid] > timeout:
            del lastseen[trackid]
            lastposition.pop(trackid , None)
            lastbox.pop(trackid , None)
            lastclass.pop(trackid , None)
            speedsmooth.pop(trackid , None)
            trail.pop(trackid , None)

    currenttime = time.time()
    fps = fps * 0.9 + (1 / max(currenttime - prevtime , 1e-6)) * 0.1
    prevtime = currenttime
    runtime = int(currenttime - starttime)

    y = 30
    for line in [
        f"Device: {devicename}" ,
        f"FPS: {fps:.1f}" ,
        f"Tracked: {trackedcount}" ,
        f"Runtime: {runtime}s" ,
    ]:
        text(frame , line , 15 , y)
        y = y + 26
    y = y + 6
    for name , count in sorted(classcounts.items()):
        text(frame , f"{name}: {count}" , 15 , y , (0 , 255 , 255))
        y = y + 22

    cv2.imshow("Drone AI Tracker" , frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    elapsed = time.time() - loopstart
    sleepneeded = frameduration - elapsed
    if sleepneeded > 0:
        time.sleep(sleepneeded)

stream.stop()
cv2.destroyAllWindows()
