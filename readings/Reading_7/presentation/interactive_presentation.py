"""
interactive_presentation.py - Reading 7 as a live, interactive Python slideshow.

    cd readings/Reading_7
    python presentation/interactive_presentation.py              # windowed
    python presentation/interactive_presentation.py --fullscreen

Keys: Right / Space / PageDown next · Left / PageUp back · F fullscreen · Esc quit.
Each slide lists its own interactive keys in the bar at the bottom.

The demos on the slides run the code in demos/ live:
    Article 1  Gorilla              BM25 word heatmap, API builder + AST trees, docs update
    Article 2  VoxPoser             3D value surface, planner lab (greedy vs A*, weights, the trap)
    Article 3  Language to Rewards  template filling, reward landscape + 256 sampled futures, corrections
    Article 4  Kine2Go              data factory, IK playground, reward kernels, rollout coverage
Numbers about my demos come from results/summary.json (python evaluate.py);
numbers from the papers are labelled as such on the slides.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from engine import run_cli  # noqa: E402
from slides_closing import ConnectionsSlide, QuizSlide, TakeawaySlide  # noqa: E402
from slides_gorilla import (GorillaASTSlide, GorillaIdeaSlide, GorillaResultsSlide,  # noqa: E402
                            GorillaRetrieverSlide)
from slides_kine2go import KineDatasetSlide, KineIdeaSlide, KineIKSlide, KineRewardSlide  # noqa: E402
from slides_l2r import L2RIdeaSlide, L2RLabSlide, L2RResultsSlide  # noqa: E402
from slides_overview import SubwaySlide, TitleSlide  # noqa: E402
from slides_voxposer import VoxIdeaSlide, VoxLabSlide, VoxResultsSlide  # noqa: E402

SLIDES = [TitleSlide, SubwaySlide,
          GorillaIdeaSlide, GorillaRetrieverSlide, GorillaASTSlide, GorillaResultsSlide,
          VoxIdeaSlide, VoxLabSlide, VoxResultsSlide,
          L2RIdeaSlide, L2RLabSlide, L2RResultsSlide,
          KineIdeaSlide, KineIKSlide, KineRewardSlide, KineDatasetSlide,
          ConnectionsSlide, QuizSlide, TakeawaySlide]


def main():
    path = os.path.join(ROOT, "results", "summary.json")
    if not os.path.exists(path):
        sys.exit("results/summary.json not found - run `python evaluate.py` first.")
    with open(path) as f:
        summary = json.load(f)
    run_cli(SLIDES, "ENG 6010 · Reading 7 · From Words to Motors", __doc__, context={"summary": summary})


if __name__ == "__main__":
    main()
