# Import numpy and OpenCV
import numpy as np
import cv2

SMOOTHING_RADIUS = 500
WINDOW_NAME = 'Video Stab'
preview_time = 5

def update_radius(x):
    global SMOOTHING_RADIUS
    SMOOTHING_RADIUS = x

def update_range(x):
    global preview_time
    preview_time = x

def create_controls():
    cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar('Radius', WINDOW_NAME, SMOOTHING_RADIUS, 1000, update_radius)
    switch = 'Preview Time:'
    cv2.createTrackbar(switch, WINDOW_NAME, preview_time, 10, update_range)

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size

    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]

    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)

    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)

    frame = cv2.warpAffine(frame, T, (s[1], s[0]))

    return frame


def get_transforms(cap, n_frames):
    # First frame setup
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Frames Transforms
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        # Detect good features in frame i-1
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # Read frame i
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Get optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # Filter valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # Find transformation matrix & save transforms
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        dx = m[0, 2]  # Tx
        dy = m[1, 2]  # Ty
        da = np.arctan2(m[1, 0], m[0, 0])  # angle
        transforms[i] = [dx, dy, da]  # Save transforms

        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    return transforms

def compute_and_save(cap, transforms, n_frames, out):
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory)

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Write n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        if (frame_out.shape[1] > 1920):
            frame_out = cv2.resize(frame_out, (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)));
        out.write(frame_out)


if __name__ == '__main__':

    # Generate controls
    # create_controls()

    # Video read setup
    cap = cv2.VideoCapture('video1.mp4')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Video read setup
    for i in range(500, 1101, 500):

        SMOOTHING_RADIUS = i
        out = cv2.VideoWriter('video_out' + str(i) + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (2 * w, h))
        cap = cv2.VideoCapture('video1.mp4')
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        transforms = get_transforms(cap, n_frames)
        compute_and_save(cap, transforms, n_frames, out)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
