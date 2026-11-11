"""
interactive_presentation.py - Reading 7 as a live, interactive Python slideshow.

    cd readings/Reading_7
    python presentation/interactive_presentation.py              # windowed
    python presentation/interactive_presentation.py --fullscreen

Keys: Right / Space / PageDown next · Left / PageUp back · F fullscreen · Esc quit.
Each slide lists its own interactive keys in the bar at the bottom.

The demos on the slides run the code in demos/ live:
    Article 1  Gorilla              BM25 retrieval, AST hallucination check, docs update
    Article 2  VoxPoser             value maps + greedy planner + MPC replanning (drag objects)
    Article 3  Language to Rewards  reward API + predictive-sampling MPC (edit the reward live)
    Article 4  Kine2Go              retargeting with IK, the imitation reward, rollout coverage
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
from slides_intro_gorilla import (GorillaDemoSlide, GorillaIdeaSlide, GorillaResultsSlide,  # noqa: E402
                                  StackSlide, TitleSlide)
from slides_kine2go_end import (KineDatasetSlide, KineIdeaSlide, KineRetargetSlide,  # noqa: E402
                                KineRewardSlide, TakeawaySlide, ThreadsSlide)
from slides_l2r import L2RDemoSlide, L2RIdeaSlide, L2RResultsSlide  # noqa: E402
from slides_voxposer import VoxDemoSlide, VoxIdeaSlide, VoxResultsSlide  # noqa: E402

SLIDES = [TitleSlide, StackSlide,
          GorillaIdeaSlide, GorillaDemoSlide, GorillaResultsSlide,
          VoxIdeaSlide, VoxDemoSlide, VoxResultsSlide,
          L2RIdeaSlide, L2RDemoSlide, L2RResultsSlide,
          KineIdeaSlide, KineRetargetSlide, KineRewardSlide, KineDatasetSlide,
          ThreadsSlide, TakeawaySlide]


def main():
    path = os.path.join(ROOT, "results", "summary.json")
    if not os.path.exists(path):
        sys.exit("results/summary.json not found - run `python evaluate.py` first.")
    with open(path) as f:
        summary = json.load(f)
    run_cli(SLIDES, "ENG 6010 · Reading 7 · From Words to Motors", __doc__, context={"summary": summary})


if __name__ == "__main__":
    main()
