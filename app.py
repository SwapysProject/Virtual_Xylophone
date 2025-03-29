import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
import pygame
from streamlit_webrtc import webrtc_streamer
import av

os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.mixer.init()

# Set screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
XYLOPHONE_HEIGHT = 100

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Define bar dimensions
BAR_WIDTH = (SCREEN_WIDTH //1.095)//9

# Define sound files directory
SOUNDS_DIR = "sounds_xylophone"

# List of xylophone sounds
XYLOPHONE_SOUNDS = [
    "note1.wav", "note2.wav", "note3.wav", "note4.wav", "note5.wav", "note6.wav", "note7.wav"
]

# Load sounds
sounds = []
for sound_file in XYLOPHONE_SOUNDS:
    sound_path = os.path.join(SOUNDS_DIR, sound_file)
    sounds.append(pygame.mixer.Sound(sound_path))

# Define rectangles for bars
bars = []
for i in range(1,8):
    rect = pygame.Rect(i * BAR_WIDTH, 0, BAR_WIDTH, XYLOPHONE_HEIGHT)
    bars.append(rect)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Streamlit layout
st.title("Virtual Xylophone")
st.markdown("""
### Instructions:
- Ensure your camera is connected.
- Use a stick or pen to virtually play the xylophone.
- Scroll down to view the key bar.
- Press the **Run** checkbox to start.
- Press **Q** to quit.
""")

FRAME_WINDOW = st.image([])
XYLOPHONE_WINDOW = st.image([])

# Track the last bars played
last_bars = set()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_bars = set()

    # Create a blank xylophone image
    xylophone = np.zeros((XYLOPHONE_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    xylophone[:, :] = WHITE

    # Draw xylophone keys
    for i, rect in enumerate(bars):
        cv2.rectangle(xylophone, (rect.x, rect.y), (rect.x + rect.width, rect.y + rect.height), BLACK, 2)
        cv2.putText(xylophone, f"{i+1}", (rect.x + 10, XYLOPHONE_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark[8:9]:  # Index finger tip
                h, w, _ = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), -1)

                for i, rect in enumerate(bars):
                    expanded_rect = rect.inflate(30, 30)
                    if expanded_rect.collidepoint(cx, cy):
                        if i not in last_bars:
                            pygame.mixer.stop()
                            sounds[i].play()
                        current_bars.add(i)

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Highlight pressed bars
    for i in current_bars:
        cv2.rectangle(xylophone, (bars[i].x, bars[i].y), (bars[i].x + bars[i].width, bars[i].y + bars[i].height), RED, -1)
        cv2.putText(xylophone, f"{i+1}", (bars[i].x + 10, XYLOPHONE_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

    global last_bars
    last_bars = current_bars

    FRAME_WINDOW.image(img, channels="BGR")
    XYLOPHONE_WINDOW.image(xylophone, channels="BGR")

webrtc_streamer(
    key="xylophone-stream",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
