from typing import Iterator, List
import pygame
import numpy as np


class Renderer:
    '''renders frames that visualize the job scheduling simulation in 
    real time. A gantt chart is displayed, with the traces of all the 
    workers are stacked vertically. Job completions are indicated by 
    red markers, and info about the simulation is displayed in text.
    '''

    def __init__(
        self,
        num_workers: int,
        num_total_jobs: int,
        window_width: int = 400,
        window_height: int = 300,
        font_name: str = 'couriernew',
        font_size: int = 16,
        render_fps: int = 30
    ):
        self.num_workers = num_workers
        self.num_total_jobs = num_total_jobs
        self.window_width = window_width
        self.window_height = window_height
        self.font_name = font_name
        self.font_size = font_size
        self.render_fps = render_fps

        self.WORKER_RECT_H = np.ceil(self.window_height / self.num_workers)
        self.window = None
        self.clock = None



    def render_frame(
        self, 
        worker_histories: Iterator[list], 
        job_completion_times: Iterator[float],
        wall_time: float,
        avg_job_duration: float,
        num_active_jobs: int,
        num_jobs_completed: int
    ) -> None:

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self.font = pygame.font.SysFont(self.font_name, 
                                            self.font_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # draw canvas
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        self._draw_worker_histories(canvas, worker_histories, wall_time)
        self._draw_job_completion_markers(canvas, job_completion_times, wall_time)
        self.window.blit(canvas, canvas.get_rect())

        # draw text
        text_surfaces = \
            self._make_text_surfaces(wall_time, 
                                     avg_job_duration, 
                                     num_active_jobs, 
                                     num_jobs_completed)
        for text_surface, pos in text_surfaces:
            self.window.blit(text_surface, pos)

        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.render_fps)



    def close(self) -> None:
        if self.window is not None:
            pygame.image.save(self.window, "screenshot.png")
            pygame.display.quit()
            pygame.quit()



    ## internal methods

    def _draw_worker_histories(self, 
                               canvas, 
                               worker_histories, 
                               wall_time):

        for i, history in enumerate(worker_histories):
            y_rect = i * self.WORKER_RECT_H
            x_rect = 0
            for i in range(len(history)):
                t, job_id = history[i]
                if i > 0:
                    t_prev = history[i-1][0]
                    assert t_prev is not None
                else:
                    t_prev = 0

                if t is None:
                    t = wall_time

                width_ratio = (t-t_prev) / wall_time
                w_rect = np.ceil(self.window_width * width_ratio)

                if job_id == -1:
                    color = (0, 0, 0)
                else:
                    color1 = np.array((0, 100, 255))
                    color2 = np.array((2, 247, 112))
                    p = ((job_id+1) / self.num_total_jobs)
                    color = color1 + p*(color2 - color1)

                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        (x_rect, y_rect),
                        (w_rect, self.WORKER_RECT_H),
                    ),
                )

                x_rect += w_rect



    def _draw_job_completion_markers(
        self, 
        canvas, 
        job_completion_times, 
        wall_time):

        for t in job_completion_times:
            x = self.window_width * (t / wall_time)

            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    (x, 0),
                    (1, self.window_height)
                )
            )



    def _make_text_surfaces(self, 
                            wall_time, 
                            avg_job_duration, 
                            num_active_jobs, 
                            num_jobs_completed,
                            dy=20):
        wall_time = int(wall_time * 1e-3)

        surfs = [
            self.font.render(f'Wall time: {wall_time}s', False, (255,)*3),
            self.font.render(f'Avg job duration: {avg_job_duration}s', False, (255,)*3),
            self.font.render(f'Num active jobs: {num_active_jobs}', False, (255,)*3),
            self.font.render(f'Num jobs completed: {num_jobs_completed}', False, (255,)*3)
        ]

        return [(surf, (0, dy*i)) for i, surf in enumerate(surfs)]