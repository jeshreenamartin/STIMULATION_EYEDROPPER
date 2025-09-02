import cv2
import mediapipe as mp
import pyautogui  # to get mouse position

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Drawing utility
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

# Open webcam
cap = cv2.VideoCapture(0)

# Screen size (for scaling mouse to frame)
screen_w, screen_h = pyautogui.size()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw only eyes
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_LEFT_EYE, landmark_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_RIGHT_EYE, landmark_drawing_spec=drawing_spec)

            # Get center of left eye (landmark 33)
            left_eye = face_landmarks.landmark[33]
            eye_x, eye_y = int(left_eye.x * w), int(left_eye.y * h)

            # Get mouse position and scale to frame
            mouse_x, mouse_y = pyautogui.position()
            mouse_x = int(mouse_x * (w / screen_w))
            mouse_y = int(mouse_y * (h / screen_h))

            # Draw eye center
            cv2.circle(frame, (eye_x, eye_y), 5, (0, 255, 0), -1)

            # Draw dropper tip (mouse)
            cv2.circle(frame, (mouse_x, mouse_y), 8, (0, 0, 255), -1)

            # Check alignment (distance threshold)
            dist = ((eye_x - mouse_x) ** 2 + (eye_y - mouse_y) ** 2) ** 0.5
            if dist < 40:  # threshold pixels
                cv2.putText(frame, "Dropper Aligned ✅", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Aligned ❌", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eye + Dropper Alignment Simulation", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
