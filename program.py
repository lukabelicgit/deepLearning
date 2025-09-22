import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pygame
import mediapipe as mp

from utils import calculate_angle
from datetime import datetime
import csv


# =========================
# Constants & Configuration
# =========================

WINDOW_NAME = "Exercise Counter"
FONT = cv2.FONT_HERSHEY_COMPLEX

# Colors (BGR)
GREEN = (46, 204, 113)
YELLOW = (0, 255, 255)
RED = (69, 53, 220)
GRAY = (211, 211, 211)
BLACK = (30, 30, 30)
WHITE = (255, 255, 255)

COUNTDOWN_SECONDS = 5
PROGRESS_BAR_WIDTH = 50
PROGRESS_BAR_HEIGHT_RATIO = 0.8

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
POSE = mp_pose.Pose()

POSE_LM = mp_pose.PoseLandmark


# Draw body connections
POSE_CONNECTIONS_EXCLUDING_FACE = [
    # Upper body
    (POSE_LM.LEFT_SHOULDER.value, POSE_LM.RIGHT_SHOULDER.value),
    (POSE_LM.LEFT_SHOULDER.value, POSE_LM.LEFT_ELBOW.value),
    (POSE_LM.LEFT_ELBOW.value, POSE_LM.LEFT_WRIST.value),
    (POSE_LM.RIGHT_SHOULDER.value, POSE_LM.RIGHT_ELBOW.value),
    (POSE_LM.RIGHT_ELBOW.value, POSE_LM.RIGHT_WRIST.value),
    # Lower body
    (POSE_LM.LEFT_HIP.value, POSE_LM.RIGHT_HIP.value),
    (POSE_LM.LEFT_HIP.value, POSE_LM.LEFT_KNEE.value),
    (POSE_LM.LEFT_KNEE.value, POSE_LM.LEFT_ANKLE.value),
    (POSE_LM.RIGHT_HIP.value, POSE_LM.RIGHT_KNEE.value),
    (POSE_LM.RIGHT_KNEE.value, POSE_LM.RIGHT_ANKLE.value),
    (POSE_LM.LEFT_ANKLE.value, POSE_LM.LEFT_HEEL.value),
    (POSE_LM.LEFT_HEEL.value, POSE_LM.LEFT_FOOT_INDEX.value),
    (POSE_LM.RIGHT_ANKLE.value, POSE_LM.RIGHT_HEEL.value),
    (POSE_LM.RIGHT_HEEL.value, POSE_LM.RIGHT_FOOT_INDEX.value),
]


class Exercise(str, Enum):
    CURLS = "curls"
    SQUATS = "squats"
    PUSHUPS = "push-ups"
    SITUPS = "sit-ups"


class GoalType(str, Enum):
    TIME = "time"
    REPS = "reps"


# Angle limits per exercise
ANGLE_LIMITS = {
    Exercise.CURLS: {"min_angle": 40, "max_angle": 150},
    Exercise.SQUATS: {"min_angle": 40, "max_angle": 150},
    Exercise.PUSHUPS: {"min_angle": 90, "max_angle": 160},
    Exercise.SITUPS: {"min_angle": 40, "max_angle": 160},
}


@dataclass
class SessionState:
    counter: int = 0
    stage: Optional[str] = None  # "up" or "down"
    start_time: Optional[float] = None
    exercise_time: float = 0.0
    exercise: Optional[Exercise] = None
    weights: float = 0.0

    repetition_times: List[float] = field(default_factory=list)
    last_rep_time: Optional[float] = None

    goal_type: Optional[GoalType] = None
    goal_value: int = 0  # seconds for time, reps for reps
    finished_input: bool = False
    workout_running: bool = True

    angles_and_times: List[List[Tuple[float, float]]] = field(default_factory=list)
    rep_angles_with_times: List[Tuple[float, float]] = field(default_factory=list)

    min_angle: int = 0
    max_angle: int = 0


# =========================
# Utility helpers
# =========================

def type_to_suffix(goal_type: Optional[GoalType]) -> str:
    if goal_type == GoalType.TIME:
        return "s"
    if goal_type == GoalType.REPS:
        return "reps"
    return ""


def clamp_goal(value: int) -> int:
    return max(0, value)


def reduce_angles_to_n(angles_with_times: List[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    """Downsample a list of (angle, time) to n evenly spaced samples."""
    num_angles = len(angles_with_times)
    if num_angles <= n:
        return angles_with_times
    indices = np.linspace(0, num_angles - 1, n, dtype=int)
    return [angles_with_times[i] for i in indices]


def get_landmark_triplet(ex: Exercise, lm) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Return (A, B, C) points used for angle calculation for the given exercise.
    Each point is (x, y) in normalized coordinates.
    """
    if ex == Exercise.CURLS:
        a = lm[POSE_LM.RIGHT_SHOULDER.value]
        b = lm[POSE_LM.RIGHT_ELBOW.value]
        c = lm[POSE_LM.RIGHT_WRIST.value]
    elif ex == Exercise.SQUATS:
        a = lm[POSE_LM.LEFT_HIP.value]
        b = lm[POSE_LM.LEFT_KNEE.value]
        c = lm[POSE_LM.LEFT_ANKLE.value]
    elif ex == Exercise.PUSHUPS:
        a = lm[POSE_LM.RIGHT_SHOULDER.value]
        b = lm[POSE_LM.RIGHT_ELBOW.value]
        c = lm[POSE_LM.RIGHT_WRIST.value]
    elif ex == Exercise.SITUPS:
        a = lm[POSE_LM.LEFT_SHOULDER.value]
        b = lm[POSE_LM.LEFT_HIP.value]
        c = lm[POSE_LM.LEFT_KNEE.value]
    else:
        raise ValueError("Unsupported exercise")
    return (a.x, a.y), (b.x, b.y), (c.x, c.y)


# =========================
# Drawing / UI helpers
# =========================

def put_text(img, text, org, scale=1, color=WHITE, thick=2):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_progress_bar(img: np.ndarray, progress_pct: float, stage: Optional[str]) -> None:
    """Draw a vertical progress bar on the right side of the frame."""
    h, w, _ = img.shape
    bar_h = int(h * PROGRESS_BAR_HEIGHT_RATIO)
    bar_x = w - PROGRESS_BAR_WIDTH - 20
    bar_y = (h - bar_h) // 2

    progress_pct = int(np.clip(progress_pct, 0, 100))
    filled_h = int((progress_pct / 100.0) * bar_h)

    # Color logic
    if progress_pct == 100:
        color = RED
    elif progress_pct == 0:
        color = GREEN
    else:
        color = RED if stage == "up" else GREEN

    # Background
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + PROGRESS_BAR_WIDTH, bar_y + bar_h), (200, 200, 200), -1)
    # Filled part
    cv2.rectangle(
        img,
        (bar_x, bar_y + (bar_h - filled_h)),
        (bar_x + PROGRESS_BAR_WIDTH, bar_y + bar_h),
        color,
        -1,
    )

    # Label
    text = f"{progress_pct}%"
    (tw, th), _ = cv2.getTextSize(text, FONT, 1, 2)
    tx = bar_x + (PROGRESS_BAR_WIDTH - tw) // 2
    ty = bar_y + bar_h + 30
    put_text(img, text, (tx, ty), scale=1, color=color, thick=2)


def draw_exercise_menu(img: np.ndarray, selected: Optional[Exercise]) -> None:
    title_c = YELLOW if not selected else GRAY
    item_c = WHITE if not selected else GRAY
    put_text(img, "Select Exercise Type:", (10, 50), 1, title_c)
    put_text(img, "1: Curls", (10, 100), 1, item_c)
    put_text(img, "2: Squats", (10, 150), 1, item_c)
    put_text(img, "3: Push-ups", (10, 200), 1, item_c)
    put_text(img, "4: Sit-ups", (10, 250), 1, item_c)
    put_text(img, "Press 1-4 to select exercise", (10, 300), 1, item_c)


def draw_goal_type_menu(img: np.ndarray, selected: Optional[GoalType]) -> None:
    title_c = YELLOW if not selected else GRAY
    line_c = WHITE if not selected else GRAY
    put_text(img, "Select Goal Type:", (10, 350), 1, title_c)
    put_text(img, "Press T for Time or R for Reps", (10, 400), 1, line_c)


def draw_goal_value_menu(img: np.ndarray, goal_value: int, goal_type: Optional[GoalType]) -> None:
    put_text(img, f"Current goal = {goal_value} {type_to_suffix(goal_type)}", (10, 450), 1, YELLOW)
    y = 500
    step = 50
    lines = [
        "Press 1 to increase by 1",
        "Press 2 to decrease by 1",
        "Press 3 to increase by 5",
        "Press 4 to decrease by 5",
        "Press 5 to increase by 10",
        "Press 6 to decrease by 10",
        "Press F to finish setting goal",
    ]
    for i, line in enumerate(lines):
        put_text(img, line, (10, y + i * step), 1, WHITE)


def draw_hud(img: np.ndarray, state: SessionState) -> None:
    put_text(img, f"Count: {state.counter}", (10, 50), 1, GREEN)
    put_text(img, f"Time: {int(state.exercise_time)}s", (10, 100), 1, GREEN)
    put_text(img, f"Goal: {state.goal_value} {type_to_suffix(state.goal_type)}", (10, 150), 1, RED)


# =========================
# Sounds
# =========================

def init_audio():
    pygame.mixer.init()
    count_snd = pygame.mixer.Sound("count.wav")
    start_snd = pygame.mixer.Sound("start.wav")
    return count_snd, start_snd


# =========================
# Logic
# =========================

def handle_pre_session_input(img: np.ndarray, state: SessionState) -> None:
    """Render menus and handle keypresses until state.finished_input is True."""
    draw_exercise_menu(img, state.exercise)
    if state.exercise is None:
        return

    draw_goal_type_menu(img, state.goal_type)
    if state.goal_type is None:
        return

    draw_goal_value_menu(img, state.goal_value, state.goal_type)


def apply_key_for_pre_session(key: int, state: SessionState) -> None:
    """Update state based on pre-session key events."""
    if state.exercise is None:
        if key == ord("1"):
            state.exercise = Exercise.CURLS
            print("You selected Curls.")
        elif key == ord("2"):
            state.exercise = Exercise.SQUATS
            print("You selected Squats.")
        elif key == ord("3"):
            state.exercise = Exercise.PUSHUPS
            print("You selected Push-ups.")
        elif key == ord("4"):
            state.exercise = Exercise.SITUPS
            print("You selected Sit-ups.")
        return

    if state.goal_type is None:
        if key == ord("t"):
            state.goal_type = GoalType.TIME
            state.goal_value = 30
            print("You selected Time goal: 30 seconds.")
        elif key == ord("r"):
            state.goal_type = GoalType.REPS
            state.goal_value = 10
            print("You selected Rep goal: 10 reps.")
        return

    # Adjust goal value
    if key == ord("1"):
        state.goal_value = clamp_goal(state.goal_value + 1)
    elif key == ord("2"):
        state.goal_value = clamp_goal(state.goal_value - 1)
    elif key == ord("3"):
        state.goal_value = clamp_goal(state.goal_value + 5)
    elif key == ord("4"):
        state.goal_value = clamp_goal(state.goal_value - 5)
    elif key == ord("5"):
        state.goal_value = clamp_goal(state.goal_value + 10)
    elif key == ord("6"):
        state.goal_value = clamp_goal(state.goal_value - 10)
    elif key in (ord("f"), ord("F")):
        state.finished_input = True
        # Set angle thresholds
        limits = ANGLE_LIMITS.get(state.exercise)
        if not limits:
            print("Error: Exercise type not found.")
            state.workout_running = False
            return
        state.min_angle = limits["min_angle"]
        state.max_angle = limits["max_angle"]
        print(f"Min angle: {state.min_angle}, Max angle: {state.max_angle}")


def countdown_overlay(cap: cv2.VideoCapture, seconds: int, count_sound: pygame.mixer.Sound) -> bool:
    """Blocking countdown with semi-transparent overlay. Returns False if frame grab fails."""
    t0 = time.time()
    remaining = seconds
    while remaining > 0:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to grab frame during countdown.")
            return False

        overlay = frame.copy()
        h, w, _ = frame.shape
        font_scale = 3
        thickness = 10

        text = f"Starting in: {remaining}"
        (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
        x = int((w - tw) / 2)
        y = int((h + th) / 2)

        # Darken background
        cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        put_text(frame, text, (x, y), font_scale, YELLOW, thickness)

        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

        if time.time() - t0 >= 1:
            remaining -= 1
            count_sound.play()
            t0 = time.time()
    return True


def process_exercise_frame(
    state: SessionState,
    landmarks,
    frame_bgr: np.ndarray,
    count_sound: pygame.mixer.Sound,
) -> None:
    """
    Compute angle, update stage/counter, draw angle/progress, and accumulate rep timings.
    """
    assert state.exercise is not None

    # Landmark triplet and angle
    p1, p2, p3 = get_landmark_triplet(state.exercise, landmarks)
    angle = calculate_angle(p1, p2, p3)

    # Angle label near the middle point (p2)
    elbow_xy = tuple(np.multiply(p2[:2], [frame_bgr.shape[1], frame_bgr.shape[0]]).astype(int))
    put_text(frame_bgr, str(int(angle)), elbow_xy, 1, WHITE, 2)

    # Record angle over time for current rep
    state.rep_angles_with_times.append((angle, state.exercise_time))

    # Stage/counter logic
    if angle > state.max_angle:
        state.stage = "down"
    if angle < state.min_angle and state.stage == "down":
        state.stage = "up"
        state.counter += 1
        count_sound.play()
        # Store a down-sampled snapshot of the rep angle curve
        state.angles_and_times.append(reduce_angles_to_n(state.rep_angles_with_times, 10))
        state.rep_angles_with_times = []

        # Goal check for reps
        if state.goal_type == GoalType.REPS and state.counter >= state.goal_value:
            state.workout_running = False

        # Time between reps
        now = time.time()
        if state.last_rep_time is not None:
            dt = now - state.last_rep_time
            state.repetition_times.append(dt)
            print(f"Time between repetitions: {dt:.2f} seconds")
        state.last_rep_time = now
        print(f"Rep count: {state.counter}")

    # Progress
    progress = (state.max_angle - angle) / max(1, (state.max_angle - state.min_angle)) * 100.0
    progress = float(np.clip(progress, 0, 100))
    draw_progress_bar(frame_bgr, progress, state.stage)


def save_results_csv(
    path: str,
    date_str: str,
    exercise: Exercise,
    reps: int,
    repetition_times: List[float],
    angles_and_times: List[List[Tuple[float, float]]],
) -> None:
    """Append session results to CSV with header if file is empty."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        f.seek(0, 2)
        if f.tell() == 0:
            writer.writerow(["Date", "Exercise", "Reps", "Time Between Reps", "Angles", "Times"])

        # Flatten and format
        flat = [item for sub in angles_and_times for item in sub]
        angles = [str(round(a)) for a, _ in flat]
        times = [str(round(t, 2)) for _, t in flat]
        time_str = ",".join(f"{round(x, 2)}" for x in repetition_times)

        writer.writerow([date_str, exercise.value, reps, time_str, ",".join(angles), ",".join(times)])


# =========================
# Main
# =========================

def main():
    # Video
    cap = cv2.VideoCapture(0)

    # Audio
    count_sound, start_sound = init_audio()

    state = SessionState()

    while state.workout_running:
        ok, frame = cap.read()
        if not ok:
            print("Error: Failed to grab frame.")
            break

        # Pose inference requires RGB input
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = POSE.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr.flags.writeable = True

        # Before start: menus + inputs
        if state.start_time is None:
            handle_pre_session_input(image_bgr, state)
            cv2.imshow(WINDOW_NAME, image_bgr)
            key = cv2.waitKey(1) & 0xFF

            # Quit early
            if key == ord("q"):
                print("Quitting...")
                break

            # Handle menu input
            apply_key_for_pre_session(key, state)

            # If setup finished, do countdown and start session
            if state.finished_input:
                if not countdown_overlay(cap, COUNTDOWN_SECONDS, count_sound):
                    break
                state.start_time = time.time()
                state.last_rep_time = state.start_time
                print("Exercise started!")
                start_sound.play()
            continue

        # Session running
        state.exercise_time = time.time() - state.start_time

        # Time goal reached?
        if state.goal_type == GoalType.TIME and state.exercise_time >= state.goal_value:
            state.workout_running = False

        try:
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Remaining text (time or reps)
                if state.goal_type == GoalType.REPS:
                    remaining = max(0, state.goal_value - state.counter)
                else:
                    remaining = max(0.0, state.goal_value - state.exercise_time)

                suffix = type_to_suffix(state.goal_type)
                put_text(image_bgr, f"Remaining: {remaining:.2f}{suffix}", (550, 100), 1.5, GREEN, 3)

                process_exercise_frame(state, lm, image_bgr, count_sound)

                # Landmarks
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    POSE_CONNECTIONS_EXCLUDING_FACE,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )

        except Exception as e:
            print(f"Error in pose estimation: {e}")

        # HUD + display
        draw_hud(image_bgr, state)
        cv2.imshow(WINDOW_NAME, image_bgr)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Post-session
    print(state.repetition_times)
    print(state.angles_and_times)

    if not state.workout_running:  # finished by goal
        today = datetime.today().strftime("%Y-%m-%d")
        save_results_csv(
            path="exercise_data.csv",
            date_str=today,
            exercise=state.exercise or Exercise.CURLS,
            reps=state.counter,
            repetition_times=state.repetition_times,
            angles_and_times=state.angles_and_times,
        )
        print("Data saved")

    print("WORKOUT FINISHED :)")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
