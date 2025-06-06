import rerun as rr
import torch

rr.init("rr_viz")
rr.connect_tcp()


def circle_scanner_visualizer(distances: torch.Tensor) -> None:
    rr.log("circle_scanner_visualizer", rr.BarChart(distances[0]))
