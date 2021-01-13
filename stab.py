import numpy as np
import cv2
import matplotlib.pyplot as plt

SMOOTHING_RADIUS = None
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


def fix_rotation(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)

    frame = cv2.warpAffine(frame, T, (s[1], s[0]))

    return frame


def get_transforms(cap, n_frames, out):

    # get first frame
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # transforms array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        # features frame i-1
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        # frame i
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # get optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        # get only valid points
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        # get transformation matrix
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        dx = m[0, 2]  # Tx
        dy = m[1, 2]  # Ty
        da = np.arctan2(m[1, 0], m[0, 0])  # angle
        transforms[i] = [dx, dy, da]  # add transform i to transforms array

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

        features_prev = np.int0(prev_pts)
        features_cur = np.int0(curr_pts)

        frame_feat = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)

        for feat_prev, feat_cur in zip(features_prev, features_cur):
            x_feat_prev, y_feat_prev = feat_prev.ravel()
            cv2.circle(frame_feat, (x_feat_prev, y_feat_prev), 2, (0, 0, 255), -1)
            x_feat_cur, y_feat_cur = feat_cur.ravel()
            cv2.circle(frame_feat, (x_feat_cur, y_feat_cur), 2, (255, 0, 0), -1)

        out.write(frame_feat)

        prev_gray = curr_gray
    out.release()

    return transforms

def compute_and_save(cap, transforms, n_frames, out):

    # get trajectory using cumulative sum of transformations and smooth evrything out
    trajectory = np.cumsum(transforms, axis=0)
    difference = smooth(trajectory) - trajectory
    transforms_smooth = transforms + difference

    for axis in range(trajectory.shape[1]):
        y = trajectory[:, axis].tolist()
        y1 = [x * -1 for x in  transforms_smooth[:, axis].tolist()]
        x = list(range(trajectory.shape[0]))
        plt.plot(x, y, label=("axis" + str(axis)))
        plt.plot(x, y1, label=("smooth_axis" + str(axis)))
    plt.legend()

    # reset stream to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # process and save n_frames-1 transformed frames
    for i in range(n_frames - 2):
        # next frame
        success, frame = cap.read()
        if not success:
            break

        # transform i for frame i from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # build up transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # apply transform to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w, h))
        frame_stabilized = fix_rotation(frame_stabilized)

        # viewport
        frame_out = cv2.vconcat([frame, frame_stabilized])
        frame_out = cv2.resize(frame_out, (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)))
        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(1)

        # save result
        out.write(frame_stabilized)

    out.release()

if __name__ == '__main__':


    for i in range(50, 51, 50):

        SMOOTHING_RADIUS = i

        cap = cv2.VideoCapture('video1.mp4')
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frmt = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('video_out' + str(i) + '.avi', frmt, fps, (w, h))
        out_features = cv2.VideoWriter('video_out_features' + str(i) + '.avi', frmt, fps, (w, h))

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        transforms = get_transforms(cap, n_frames, out_features)
        compute_and_save(cap, transforms, n_frames, out)

        cap.release()
    cv2.destroyAllWindows()
    plt.show()


