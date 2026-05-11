import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# --- AI SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

st.set_page_config(page_title="GlowUp AI Pro", page_icon="💎", layout="wide")
st.title("💎 GlowUp AI: Total Analysis")

# --- FUNCTIONS ---
def get_face_shape(landmarks, w, h):
    # Get specific points (standard MediaPipe indices)
    top = landmarks[10].y * h
    bottom = landmarks[152].y * h
    left = landmarks[234].x * w
    right = landmarks[454].x * w
    jaw_left = landmarks[58].x * w
    jaw_right = landmarks[288].x * w

    face_height = bottom - top
    face_width = right - left
    jaw_width = jaw_right - jaw_left
    ratio = face_height / face_width

    if ratio > 1.5:
        return "Oblong", "Square or Round frames"
    elif ratio < 1.2:
        return "Round", "Rectangular or Angular frames"
    elif jaw_width / face_width > 0.85:
        return "Square", "Round or Oval frames"
    else:
        return "Oval", "Any frame style (Lucky you!)"

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Your Profile")
skin_type = st.sidebar.selectbox("Skin Type", ["Oily", "Dry", "Combination", "Sensitive"])
activity_level = st.sidebar.slider("Daily Activity (1-10)", 1, 10, 7)

# --- MAIN APP ---
uploaded_file = st.camera_input("Take a selfie for analysis")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        shape, glasses = get_face_shape(landmarks, w, h)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img_rgb, use_container_width=True)
            st.success(f"Detected Face Shape: **{shape}**")
            st.info(f"Recommended Glasses: **{glasses}**")

        with col2:
            st.subheader("Personalized Routine")
            
            # --- ROUTINE GENERATOR LOGIC ---
            st.write("### ☀️ Morning")
            if skin_type == "Oily":
                st.write("- **Wash:** Salicylic Acid Cleanser (CeraVe SA)")
                st.write("- **Moisturize:** Lightweight Oil-free Gel")
            else:
                st.write("- **Wash:** Gentle Cream Cleanser (Cetaphil)")
                st.write("- **Moisturize:** Hyaluronic Acid Cream")
            
            st.write("### 🌙 Night (Athlete Recovery)")
            if activity_level > 5:
                st.write("- **Double Cleanse:** Remove sweat/grime with a foaming wash.")
                st.write("- **Repair:** Use a Night Cream with Ceramides to fix the skin barrier.")
            
            st.write("### 👓 Style Tips")
            st.write(f"- Since you have a **{shape}** face, avoid frames that mimic your shape. Contrast is key.")
            st.write("- **Hair Tip:** For your textured fringe, use sea salt spray for volume.")

        # AI Mesh Visualization
        with st.expander("See AI Landmark Map"):
            annotated_img = img_rgb.copy()
            for lm in landmarks:
                cv2.circle(annotated_img, (int(lm.x*w), int(lm.y*h)), 1, (0, 255, 0), -1)
            st.image(annotated_img, use_container_width=True)
    else:
        st.error("Face not found. Please try again in better light.")