import numpy as np
import random
import cv2

# Minimal Tetris-like environment with rendering to image.
# Grid: height x width (e.g., 20 x 10)
# Actions: 0=left,1=right,2=rotate,3=soft_drop,4=hard_drop,5=no_op
# Observation: image frame (H,W,C) when render_frame True

TETROMINOES = {
    'I': [[[1,1,1,1]],
          [[1],[1],[1],[1]]],
    'O': [[[1,1],
           [1,1]]],
    'T': [[[0,1,0],
           [1,1,1]],
          [[1,0],
           [1,1],
           [1,0]],
          [[1,1,1],
           [0,1,0]],
          [[0,1],
           [1,1],
           [0,1]]],
    'S': [[[0,1,1],
           [1,1,0]],
          [[1,0],
           [1,1],
           [0,1]]],
    'Z': [[[1,1,0],
           [0,1,1]],
          [[0,1],
           [1,1],
           [1,0]]],
    'J': [[[1,0,0],
           [1,1,1]],
          [[1,1],
           [1,0],
           [1,0]],
          [[1,1,1],
           [0,0,1]],
          [[0,1],
           [0,1],
           [1,1]]],
    'L': [[[0,0,1],
           [1,1,1]],
          [[1,0],
           [1,0],
           [1,1]],
          [[1,1,1],
           [1,0,0]],
          [[1,1],
           [0,1],
           [0,1]]],
}

PIECE_NAMES = list(TETROMINOES.keys())

class TetrisEnv:
    def __init__(self, height=20, width=10, seed=None, render_frame=True, image_size=(84,84), channels=3):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.score = 0
        self.done = False
        self.current = None
        self.current_pos = (0, 0)
        self.rng = random.Random(seed)
        self.render_frame = render_frame
        self.image_size = image_size
        self.channels = channels
        self.spawn_new_piece()

    def spawn_new_piece(self):
        name = self.rng.choice(PIECE_NAMES)
        rots = TETROMINOES[name]
        rot_idx = 0
        shape = np.array(rots[rot_idx], dtype=np.uint8)
        w, h = shape.shape[1], shape.shape[0]
        x = (self.width - w) // 2
        y = 0
        self.current = {"name": name, "rot_idx": rot_idx, "shape": shape}
        self.current_pos = (y, x)
        if self._collides(shape, y, x):
            self.done = True

    def _collides(self, shape, top, left):
        sh_h, sh_w = shape.shape
        if left < 0 or left + sh_w > self.width or top + sh_h > self.height:
            return True
        sub = self.grid[top:top+sh_h, left:left+sh_w]
        return np.any((sub + shape) > 1)

    def _place_piece(self, shape, top, left):
        sh_h, sh_w = shape.shape
        self.grid[top:top+sh_h, left:left+sh_w] = np.maximum(self.grid[top:top+sh_h, left:left+sh_w], shape)

    def _clear_lines(self):
        full = np.all(self.grid == 1, axis=1)
        num = int(np.sum(full))
        if num > 0:
            self.grid = np.vstack([self.grid[~full], np.zeros((num, self.width), dtype=np.uint8)])
        return num

    def reset(self):
        self.grid[:] = 0
        self.score = 0
        self.done = False
        self.spawn_new_piece()
        return self.render()

    def render(self):
        # Render the grid + current piece into an RGB image.
        cell_h = self.image_size[0] // self.height
        cell_w = self.image_size[1] // self.width
        img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        # Draw placed blocks
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c]:
                    y0 = r * cell_h
                    x0 = c * cell_w
                    img[y0:y0+cell_h, x0:x0+cell_w] = (200, 200, 200)
        # Draw current piece
        shape = self.current["shape"]
        y, x = self.current_pos
        sh_h, sh_w = shape.shape
        for i in range(sh_h):
            for j in range(sh_w):
                if shape[i, j]:
                    gr = y + i
                    gc = x + j
                    if 0 <= gr < self.height and 0 <= gc < self.width:
                        y0 = gr * cell_h
                        x0 = gc * cell_w
                        img[y0:y0+cell_h, x0:x0+cell_w] = (0, 200, 200)
        # Resize strictly to image_size in case of integer division issues
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        if self.channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None]
        return img.astype(np.uint8)

    def step(self, action):
        if self.done:
            return self.render(), 0.0, True, {}
        y, x = self.current_pos
        shape = self.current["shape"]
        reward = 0.0

        if action == 0:  # left
            nx = x - 1
            if not self._collides(shape, y, nx):
                x = nx
        elif action == 1:  # right
            nx = x + 1
            if not self._collides(shape, y, nx):
                x = nx
        elif action == 2:  # rotate
            rots = TETROMINOES[self.current["name"]]
            nidx = (self.current["rot_idx"] + 1) % len(rots)
            nshape = np.array(rots[nidx], dtype=np.uint8)
            if not self._collides(nshape, y, x):
                self.current["rot_idx"] = nidx
                self.current["shape"] = nshape
                shape = nshape
        elif action == 3:  # soft drop
            if not self._collides(shape, y+1, x):
                y += 1
            else:
                self._place_piece(shape, y, x)
                cleared = self._clear_lines()
                reward += 1.0 + cleared * 10.0
                self.spawn_new_piece()
        elif action == 4:  # hard drop
            while not self._collides(shape, y+1, x):
                y += 1
            self._place_piece(shape, y, x)
            cleared = self._clear_lines()
            reward += 1.0 + cleared * 10.0
            self.spawn_new_piece()
        elif action == 5:
            # no op; gravity applies
            if not self._collides(shape, y+1, x):
                y += 1
            else:
                self._place_piece(shape, y, x)
                cleared = self._clear_lines()
                reward += 1.0 + cleared * 10.0
                self.spawn_new_piece()

        self.current_pos = (y, x)
        if self.done:
            reward -= 10.0
        return self.render(), float(reward), bool(self.done), {}

    @property
    def action_space_n(self):
        return 6