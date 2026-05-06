from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import cv2
import face_recognition
import time

# ✅ Authenticate Google Drive
gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")  # Use saved credentials (avoids login every time)

if gauth.credentials is None:
    gauth.LocalWebserverAuth()
    gauth.SaveCredentialsFile("mycreds.txt")  # Save login for future use
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()

drive = GoogleDrive(gauth)

# ✅ Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # ✅ Detect faces before saving
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) > 0:  # Only save if a face is detected
        cv2.putText(frame, "Press 'S' to Save Face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")  # Unique filename
            filename = f"unknown_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            # ✅ Upload to Google Drive
            gfile = drive.CreateFile({'title': filename})
            gfile.SetContentFile(filename)
            gfile.Upload()
            print(f"📤 Image Uploaded to Google Drive as {filename}!")

    cv2.imshow("AI Webcam - Cloud Storage", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
