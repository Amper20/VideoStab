import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random as rng
from operator import itemgetter
from scipy.ndimage.filters import gaussian_filter1d

class Kmeans:
    def __init__(self, k=2, tol=0.001, random_state=0):
        self.k = k
        self.tol = tol
        self.max_iter = 200
        self.random_state = 10

    def fit(self, data):

        self.cluster_centers_ = {}

        for i in range(self.k):
            self.cluster_centers_[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.cluster_centers_[centroid]) for centroid in self.cluster_centers_]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.cluster_centers_)

            for classification in self.classifications:
                self.cluster_centers_[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.cluster_centers_:
                original_centroid = prev_centroids[c]
                current_centroid = self.cluster_centers_[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.cluster_centers_[centroid]) for centroid in self.cluster_centers_]
        classification = distances.index(min(distances))
        return classification

max_quality_level = 100
rng.seed(42)
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


def get_transforms(cap, n_frames, out):
    # get first frame
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # transforms array
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 2):
        # features frame i-1
        prev_pts = find_shitomasi_corners(prev_gray)

        # frame i
        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_pts = find_shitomasi_corners(curr_gray)

        prev_pts, curr_pts = correlation(prev_pts, curr_pts)

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
    for i in range(3):
        transforms_smooth[:, i] = gaussian_filter1d(transforms_smooth[:, i], sigma=4)

    for axis in range(trajectory.shape[1]):
        y1 = [x * -1 for x in transforms_smooth[:, axis].tolist()]
        x = list(range(trajectory.shape[0]))
        plt.plot(x, y1, label=("transform_" + str(axis)))
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

        # viewport
        frame_out = cv2.vconcat([frame, frame_stabilized])
        frame_out = cv2.resize(frame_out, (int(frame_out.shape[1] / 2), int(frame_out.shape[0] / 2)))
        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(1)

        # save result
        out.write(frame_stabilized)

    out.release()


def reduce(point_cloud):
    kmeans = KMeans(n_clusters=15, random_state=0).fit(point_cloud)
    return kmeans.cluster_centers_


def correlation(prev, curr):

    prev_center = reduce(prev)
    curr_center = reduce(curr)
    distances = []
    for x in prev_center:
        for y in curr_center:
            distances.append([abs(x[0] - y[0]) + abs(x[1] - y[1]), x, y, 1])
    distances = sorted(distances, key=itemgetter(0))
    final_pairs = []
    for x in range(10):
        for pr in distances:
            pair = pr
            if pair[-1] == 1:
                for prx in distances:
                    if prx[1].all() == pair[1].all() or prx[2].all() == pair[2].all():
                        prx[-1] = 0
                        pr[-1] = 0
                        break
                final_pairs.append(pair)
                break
            else:
                continue
    # plt.plot(, 'bo', linestyle='None', markersize=2.0)
    # plt.plot(, 'ro', linestyle='None', markersize=2.0)
    # plt.show()

    corner_list_prev = [x[1] for x in final_pairs]
    corner_list_curr = [x[2] for x in final_pairs]

    cnt = len(corner_list_prev)

    return np.reshape(np.array([np.array(x) for x in corner_list_prev]), (cnt, 1, 2)), np.reshape(
        np.array([np.array(x) for x in corner_list_curr]), (cnt, 1, 2))


def find_shitomasi_corners(src_gray):
    corner_list = []
    block_size = 5
    aperture_size = 3
    shitomasi_quality_level = 10

    shitomasi_dst = cv2.cornerMinEigenVal(src_gray, block_size, aperture_size)
    shitomasi_min,  shitomasi_max, _, _ = cv2.minMaxLoc( shitomasi_dst)

    shitomasi_quality_level = max(shitomasi_quality_level, 1)

    for i in range(src_gray.shape[0]):
        for j in range(src_gray.shape[1]):
            if shitomasi_dst[i, j] > shitomasi_min + (shitomasi_max - shitomasi_min) * shitomasi_quality_level / max_quality_level:
                corner_list.append([j, i])

    return corner_list


if __name__ == '__main__':
    SMOOTHING_RADIUS = 50

    cap = cv2.VideoCapture('video4.mp4')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frmt = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('video_out4.avi', frmt, fps, (w, h))
    out_features = cv2.VideoWriter('video_out_features.avi', frmt, fps, (w, h))
    # for x in range(10):
    #     _, pr = cap.read()
    #     _, cr = cap.read()
    #     prev = find_harris_corners(cv2.cvtColor(pr, cv2.COLOR_BGR2GRAY))
    #     curr = find_harris_corners(cv2.cvtColor(cr, cv2.COLOR_BGR2GRAY))
    #     filter(prev, curr)
    transforms = get_transforms(cap, n_frames, out_features)
    compute_and_save(cap, transforms, n_frames, out)

    cap.release()

    cv2.destroyAllWindows()
    plt.show()
