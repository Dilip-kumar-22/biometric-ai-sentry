# Neural Fortress: Biometric AI Security 🛡️👤

A high-performance facial recognition system built on the **dlib** HOG (Histogram of Oriented Gradients) and Deep Learning models. This system performs real-time identity verification and automated entry logging.

## 🧠 Technical Architecture
The system utilizes a 128-dimension face encoding generated from a pre-trained ResNet-style neural network.
1. **Detection:** Locates human faces in the video stream.
2. **Alignment:** Normalizes the face to handle rotations/angles.
3. **Encoding:** Generates a unique mathematical fingerprint of the face.
4. **Verification:** Compares the fingerprint against the `authorized` personnel database using Euclidean distance.



## 🛠️ Technology Stack
* **AI Framework:** `face_recognition` (dlib-based)
* **Computer Vision:** OpenCV (`cv2`)
* **Numeric Processing:** NumPy
* **Language:** Python 3.12

## 🚀 Usage
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set Up Personnel:**
    * Create a folder named `authorized`.
    * Place a clear photo of yourself inside (e.g., `Dilip_Kumar.jpg`).
3.  **Deploy:**
    ```bash
    python facial_sentry.py
    ```
