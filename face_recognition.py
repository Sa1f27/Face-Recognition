import cv2
from deepface import DeepFace
import os
import numpy as np

def lkf(kf='known_faces'):
    kf_emb = []
    kn = []

    for fn in os.listdir(kf):
        if fn.endswith(('.jpg', '.jpeg', '.png')):
            fp = os.path.join(kf, fn)
            try:
                emb = DeepFace.represent(img_path=fp, model_name='Facenet', enforce_detection=False)
                if emb:
                    kf_emb.append(np.array(emb[0]['embedding']))
                    kn.append(os.path.splitext(fn)[0])
                else:
                    print(f"No emb for {fp}")
            except Exception as e:
                print(f"Err {fp}: {e}")

    return kf_emb, kn

def sr():
    f_emb, kn = lkf()
    cam = cv2.VideoCapture(0)

    while True:
        ok, frm = cam.read()

        try:
            faces = DeepFace.extract_faces(img_path=frm, enforce_detection=False, detector_backend='opencv')
            if not faces:
                print("No faces")
                continue

            for f in faces:
                fa = f['facial_area']
                try:
                    de = np.array(
                        DeepFace.represent(img_path=f['face'], model_name='Facenet', enforce_detection=False)[0][
                            'embedding'])

                    bm = None
                    sd = float("inf")

                    for k_emb, n in zip(f_emb, kn):
                        d = np.linalg.norm(k_emb - de)
                        if d < sd:
                            sd = d
                            bm = n
                    
                    lbl = bm if bm else "Unknown"
                except Exception as e:
                    lbl = "Err"
                    print(f"Err emb for det face: {e}")

                x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']
                cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frm, lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        except Exception as e:
            print(f"Err det face: {e}")

        cv2.imshow('Vid', frm)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sr()
