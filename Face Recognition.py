import cv2
import argparse

def detect_faces_from_image(image_path, output_path=None):
    img = cv2.imread(image_path)
    faces = detect_faces(img)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, f"Faces detected: {len(faces)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        cv2.imshow('Detected Faces', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_faces_from_video(video_source=0, output_path=None):
    cap = cv2.VideoCapture(video_source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    if output_path:
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frameana)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if output_path:
            out.write(frame)
        cv2.imshow('Detected Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection using Haar Cascades and OpenCV")
    parser.add_argument("--image", help="Path to the image file")
    parser.add_argument("--video", help="Path to the video file or camera index (default: 0)", type=int, default=0)
    parser.add_argument("--output", help="Path to save the processed image or video")
    args = parser.parse_args()

    if args.image:
        detect_faces_from_image(args.image, args.output)
    else:
        detect_faces_from_video(args.video, args.output)
