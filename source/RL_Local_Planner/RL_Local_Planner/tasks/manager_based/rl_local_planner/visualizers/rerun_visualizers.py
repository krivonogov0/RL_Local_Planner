import rerun as rr
import torch

rr.init("rr_viz")
rr.connect_tcp()


def circle_scanner_visualizer(frame_name: str, distances: torch.Tensor) -> None:
    """Visualizes circular scanner distance measurements using Rerun."""
    rr.log(f"circle_scanner_visualizer_{frame_name}", rr.BarChart(distances[0]))


def depth_top_view_visualizer(frame_name: str, depth_image: torch.Tensor) -> None:
    """Visualizes depth top view (privileged info) using Rerun."""
    rr.log(f"depth_top_view_visualizer_{frame_name}", rr.DepthImage(depth_image[0].cpu().numpy()))

def semantic_top_view_visualizer(frame_name: str, semantic_image: torch.Tensor) -> None:
    """Visualizes semantic top view (privileged info) using Rerun."""
    rr.log(f"semantic_top_view_visualizer_{frame_name}", rr.Image(semantic_image[0].cpu().numpy()))
