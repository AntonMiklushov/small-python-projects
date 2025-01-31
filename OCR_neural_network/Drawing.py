import pygame as pg
import numpy as np

pg.init()


class Drawing:
    WIDTH, HEIGHT, GRID_SIZE = 28 * 20, 28 * 20, 20

    def __init__(self, nn):
        self.nn = nn
        self._window = pg.display.set_mode((self.WIDTH, self.HEIGHT))  # Initialize display
        self.running = True
        self.grid = np.zeros((self.HEIGHT // self.GRID_SIZE, self.WIDTH // self.GRID_SIZE), dtype=int)  # Grid storage
        self.drawing = False

    def event_listener(self):
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.running = False
                pg.display.quit()
            elif e.type == pg.MOUSEBUTTONDOWN:
                if e.button == 1:  # Start drawing
                    self.drawing = True
                    self.fill_cell(e.pos)
                elif e.button == 3:  # Clear grid
                    self.clear_grid()
            elif e.type == pg.MOUSEMOTION and self.drawing:
                self.fill_cell(e.pos)
            elif e.type == pg.MOUSEBUTTONUP:
                if e.button == 1:  # Stop drawing
                    self.drawing = False
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_SPACE:
                    l = self.nn.forward_propagation(self.grid.reshape(1, -1).T) / 255
                    print(list(l).index(max(l)))


    def clear_grid(self):
        self.grid.fill(0)  # Clear the grid

    def fill_cell(self, mouse_pos):
        x, y = mouse_pos
        col = x // self.GRID_SIZE
        row = y // self.GRID_SIZE

        if 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]:  # Bounds check
            self.grid[row, col] = min(255, self.grid[row, col] + 50)  # Core cell intensity logic

            for drow in [-1, 0, 1]:  # Apply gradient to neighbors
                for dcol in [-1, 0, 1]:
                    nrow, ncol = row + drow, col + dcol
                    if 0 <= nrow < self.grid.shape[0] and 0 <= ncol < self.grid.shape[1]:
                        distance = ((x - (ncol * self.GRID_SIZE + self.GRID_SIZE / 2)) ** 2 +
                                    (y - (nrow * self.GRID_SIZE + self.GRID_SIZE / 2)) ** 2) ** 0.5
                        if distance < self.GRID_SIZE:
                            self.grid[nrow, ncol] += (1 - distance / self.GRID_SIZE) * 50
                            self.grid[nrow, ncol] = min(255, self.grid[nrow, ncol])  # Cap intensity

    def draw_grid(self):
        for row in range(self.grid.shape[0]):  # Render grid cells
            for col in range(self.grid.shape[1]):
                color = (self.grid[row, col],) * 3  # Grayscale based on intensity
                rect = pg.Rect(col * self.GRID_SIZE, row * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                pg.draw.rect(self._window, color, rect)

    def get_state(self):
        return self.running

    def get_grid(self):
        return self.grid

    def window_loop(self):
        self._window.fill((0, 0, 0))  # Clear screen
        self.draw_grid()
        pg.display.flip()  # Update display
        self.event_listener()
