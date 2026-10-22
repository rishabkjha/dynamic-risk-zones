"""Top-down view (OpenCV drawing, fast enough for live use): zones, tracks, candidate paths, chosen path."""
import cv2
import numpy as np

import config

ZONE_COLOURS = [(60, 60, 230), (60, 160, 240), (120, 220, 240)]     # inner red, middle orange, outer yellow (BGR)


class TopDown:
    def __init__(self, size=480, metres=8.0, centre=(3.0, 0.0), follow=False):
        self.size = size
        self.scale = size / metres          # pixels per metre
        self.centre = np.array(centre, float)
        self.follow = follow

    def px(self, xy):
        """World (x forward/up the screen, y left) -> pixel (u, v)."""
        xy = np.asarray(xy, float)
        d = xy - self.centre
        u = self.size / 2 - d[..., 1] * self.scale
        v = self.size / 2 - d[..., 0] * self.scale
        return np.stack([u, v], -1)

    def draw(self, robot_pose, goal, tracks, zones, paths=None, best=None, peds=None,
             times_to_show=(0.0, 1.0, 2.0, 3.0), title=""):
        if self.follow:
            self.centre = robot_pose[:2] + 2.0 * np.array([np.cos(robot_pose[2]), np.sin(robot_pose[2])])
        img = np.full((self.size, self.size, 3), 255, np.uint8)
        # 1 m grid
        lo = self.centre - self.size / 2 / self.scale
        hi = self.centre + self.size / 2 / self.scale
        for gx in np.arange(np.floor(lo[0]), np.ceil(hi[0]) + 1):
            p = self.px([[gx, lo[1]], [gx, hi[1]]]).astype(int)
            cv2.line(img, tuple(p[0]), tuple(p[1]), (235, 235, 235), 1)
        for gy in np.arange(np.floor(lo[1]), np.ceil(hi[1]) + 1):
            p = self.px([[lo[0], gy], [hi[0], gy]]).astype(int)
            cv2.line(img, tuple(p[0]), tuple(p[1]), (235, 235, 235), 1)

        # zones: outer levels first, several future times, fainter further ahead
        if zones is not None and not zones.empty:
            overlay = img.copy()
            for t_show in sorted(times_to_show, reverse=True):
                k = int(np.argmin(np.abs(zones.times - t_show)))
                for level in (2, 1, 0):
                    for n in range(len(zones.centers)):
                        c = self.px(zones.centers[n, k]).astype(int)
                        a, b = zones.axes[n, k, level] * self.scale
                        ang = -np.degrees(zones.theta[n]) - 90
                        cv2.ellipse(overlay, tuple(c), (max(int(a), 1), max(int(b), 1)), ang, 0, 360,
                                    ZONE_COLOURS[level], -1)
            img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
            for n in range(len(zones.centers)):          # predicted centre path
                pts = self.px(zones.centers[n]).astype(np.int32)
                cv2.polylines(img, [pts], False, (40, 40, 160), 1)

        if paths is not None:
            for p in paths[::3]:
                cv2.polylines(img, [self.px(p).astype(np.int32)], False, (210, 210, 210), 1)
            if best is not None:
                cv2.polylines(img, [self.px(paths[best]).astype(np.int32)], False, (40, 170, 40), 3)

        if peds is not None:                              # ground truth (simulation only)
            for p in peds:
                cv2.circle(img, tuple(self.px(p).astype(int)), int(0.2 * self.scale), (90, 90, 90), 1)
        for tr in tracks:
            c = tuple(self.px(tr.pos).astype(int))
            cv2.circle(img, c, 4, (0, 0, 0), -1)
            speed = float(np.linalg.norm(tr.vel))
            cv2.putText(img, f"{tr.label[:6]} {tr.id} {speed:.1f}m/s", (c[0] + 6, c[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

        g = tuple(self.px(goal).astype(int))
        cv2.drawMarker(img, g, (200, 0, 200), cv2.MARKER_STAR, 16, 2)
        r = self.px(robot_pose[:2]).astype(int)
        cv2.circle(img, tuple(r), int(config.ROBOT_RADIUS * self.scale), (200, 120, 0), 2)
        tip = self.px(robot_pose[:2] + 0.4 * np.array([np.cos(robot_pose[2]), np.sin(robot_pose[2])]))
        cv2.line(img, tuple(r), tuple(tip.astype(int)), (200, 120, 0), 2)
        if title:
            cv2.putText(img, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img
