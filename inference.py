# inference.py – OpenMV deployment (Nicla Vision)
# Garbage Classification CNN (TFLite, full int8 weights, float32 I/O)

import sensor
import image
import time
import ml

from machine import LED

#
# Settings
#
MODEL                = "custom_objects_dynamic_range.tflite"
#MODEL                = "custom_objects_float16.tflite"
#MODEL                = "custom_objects_float32.tflite"
#MODEL                = "custom_objects_int8_float_io.tflite"
#MODEL                = "custom_objects_int8.tflite"
LABELS               = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CONFIDENCE_THRESHOLD = 0.6   # minimum confidence to accept a prediction
DEBOUNCE_FRAMES      = 10    # frames a class must hold before updating the LED

# Recyclable vs landfill mapping for LED feedback
RECYCLABLE = {'cardboard', 'glass', 'metal', 'paper', 'plastic'}

led_red   = LED("LED_RED")
led_green = LED("LED_GREEN")
led_blue  = LED("LED_BLUE")

def all_leds_off():
    led_red.off()
    led_green.off()
    led_blue.off()

#
# Load model before sensor.reset() so it claims framebuffer memory first
#
net = ml.Model(MODEL, load_to_fb=True)

#
# Configure camera to match training: RGB565, 64x64
#
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)        # B&W, matches training
sensor.set_framesize(sensor.B64X64)           # native 64x64 sensor format
sensor.set_windowing((64, 64))                # crop to 64x64 input
sensor.skip_frames(time=2000)                 # let the camera settle
sensor.set_auto_whitebal(False)               # fixed white balance
sensor.set_auto_exposure(False, exposure_us=5000)  # fixed exposure

clock = time.clock()

candidate    = None   # class seen in the most recent frame (or None if low confidence)
stable_class = None   # last class that held for DEBOUNCE_FRAMES
hold_count   = 0

all_leds_off()
led_blue.on()   # blue = idle / uncertain

while True:
    clock.tick()
    img = sensor.snapshot()

    probs      = net.predict([img])[0][0].tolist()
    cls        = probs.index(max(probs))
    confidence = max(probs)
    label      = LABELS[cls] if confidence >= CONFIDENCE_THRESHOLD else None

    print("FPS:", clock.fps(), " Conf:", confidence, " Class:", LABELS[cls])

    # Debounce: require the same class for DEBOUNCE_FRAMES consecutive frames
    if label == candidate:
        hold_count += 1
    else:
        candidate  = label
        hold_count = 0

    if hold_count >= DEBOUNCE_FRAMES and candidate != stable_class:
        stable_class = candidate
        all_leds_off()
        if stable_class is None:
            led_blue.on()                    # uncertain / below threshold
        elif stable_class in RECYCLABLE:
            led_green.on()                   # recyclable material
        else:
            led_red.on()                     # non-recyclable trash
        print("Stable detection:", stable_class)
