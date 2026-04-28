import os
import cv2
import numpy as np
import mediapipe as mp


def get_face_mask(frame, bbox):
    """Create a binary mask where the face region is 1 and the background is 0."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    x, y, w, h = bbox

    x = max(0, x)
    y = max(0, y)
    w = max(0, w)
    h = max(0, h)

    mask[y:y + h, x:x + w] = 1
    return mask


def extract_features(video_path, max_duration=10, max_frames=300):
    """
    Extract physiological, motion, texture, and compression-related features
    for deepfake detection.
    """
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1
    )

    face_detector = mp_face_detection.FaceDetection(
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count == 0 or fps == 0:
        print(f"[ERROR] Invalid video: {video_path}")
        cap.release()
        return None

    frames_to_analyze = int(min(frame_count, max_duration * fps))

    if frames_to_analyze <= max_frames:
        frame_indices = np.arange(frames_to_analyze)
    else:
        frame_indices = np.linspace(
            0,
            frames_to_analyze - 1,
            max_frames,
            dtype=int
        )

    ret, prev_frame = cap.read()
    if not ret:
        print(f"[ERROR] Could not read first frame: {video_path}")
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_landmarks = None

    inconsistency_scores = []
    smoothness_scores = []
    texture_scores = []
    color_std_scores = []
    face_motion_std_scores = []
    motion_ratio_scores = []
    entropy_scores = []
    blur_scores = []
    dct_scores = []
    blockiness_scores = []

    for i in range(frames_to_analyze):
        ret, frame = cap.read()
        if not ret:
            break

        if i not in frame_indices:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        results = face_detector.process(rgb_frame)

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box

            H, W, _ = frame.shape

            x = int(bboxC.xmin * W)
            y = int(bboxC.ymin * H)
            w = int(bboxC.width * W)
            h = int(bboxC.height * H)

            x = max(0, x)
            y = max(0, y)
            w = min(w, W - x)
            h = min(h, H - y)

            if w <= 0 or h <= 0:
                continue

            face_mask = get_face_mask(frame, (x, y, w, h))
            bg_mask = 1 - face_mask

            if np.count_nonzero(face_mask) == 0 or np.count_nonzero(bg_mask) == 0:
                continue

            face_motion = np.mean(mag[face_mask == 1])
            bg_motion = np.mean(mag[bg_mask == 1])

            inconsistency = abs(face_motion - bg_motion) / (bg_motion + 1e-5)
            inconsistency_scores.append(inconsistency)

            face_motion_std_scores.append(np.std(mag[face_mask == 1]))

            motion_ratio = np.mean(flow[..., 0]) / (np.mean(flow[..., 1]) + 1e-5)
            motion_ratio_scores.append(motion_ratio)

            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture_scores.append(laplacian_var)

            skin_hue_std = np.std(hsv[..., 0][face_mask == 1])
            color_std_scores.append(skin_hue_std)

            face_gray = gray[y:y + h, x:x + w]

            if face_gray.size > 0:
                hist = cv2.calcHist(
                    [face_gray],
                    [0],
                    None,
                    [256],
                    [0, 256]
                )

                hist_prob = hist / np.sum(hist)
                hist_prob = hist_prob[hist_prob > 0]

                entropy = -np.sum(hist_prob * np.log2(hist_prob))
                entropy_scores.append(entropy)

                gx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)

                grad_mag = np.sqrt(gx ** 2 + gy ** 2)
                blur_scores.append(np.var(grad_mag))

                face_resized = cv2.resize(face_gray, (64, 64))
                dct = cv2.dct(np.float32(face_resized))
                dct_scores.append(np.mean(np.abs(dct[32:, 32:])))

                blockiness = 0.0

                for ii in range(8, face_gray.shape[0] - 8, 8):
                    blockiness += np.mean(
                        np.abs(face_gray[ii, :] - face_gray[ii - 1, :])
                    )

                for jj in range(8, face_gray.shape[1] - 8, 8):
                    blockiness += np.mean(
                        np.abs(face_gray[:, jj] - face_gray[:, jj - 1])
                    )

                denominator = (
                    face_gray.shape[0] // 8
                    + face_gray.shape[1] // 8
                    + 1e-5
                )

                blockiness /= denominator
                blockiness_scores.append(blockiness)

        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            landmarks = np.array([
                [lm.x, lm.y]
                for lm in face_landmarks.landmark
            ])

            if prev_landmarks is not None:
                displacement = np.linalg.norm(
                    landmarks - prev_landmarks,
                    axis=1
                )
                smoothness_scores.append(np.mean(displacement))

            prev_landmarks = landmarks

        prev_gray = gray.copy()

    cap.release()
    face_mesh.close()
    face_detector.close()

    return {
        "video": os.path.basename(video_path),
        "motion_inconsistency": np.mean(inconsistency_scores) if inconsistency_scores else None,
        "smoothness_mean": np.mean(smoothness_scores) if smoothness_scores else None,
        "smoothness_std": np.std(smoothness_scores) if smoothness_scores else None,
        "face_motion_std": np.mean(face_motion_std_scores) if face_motion_std_scores else None,
        "motion_ratio": np.mean(motion_ratio_scores) if motion_ratio_scores else None,
        "texture_variance": np.mean(texture_scores) if texture_scores else None,
        "skin_hue_std": np.mean(color_std_scores) if color_std_scores else None,
        "entropy_mean": np.mean(entropy_scores) if entropy_scores else None,
        "blur_variance": np.mean(blur_scores) if blur_scores else None,
        "dct_energy": np.mean(dct_scores) if dct_scores else None,
        "blockiness": np.mean(blockiness_scores) if blockiness_scores else None,
    }
