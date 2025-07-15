import logging

from typing import List, Dict, Any
from utils.beat_utils import create_beat_graph, get_beat_segment
from utils.arrangement_utils import ArrangementLengths, ArrangementPath, has_redundant_chain


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArrangementSystem:
    def __init__(self, arrangement_config):
        self.config = arrangement_config

    def calculate_path_score(self,
                             lengths: ArrangementLengths,
                             transitions: List[Dict[str, Any]],
                             metadata: Dict[str, Any],
                             ) -> float:
        """
        Calculate the score for a given arrangement path based on its lengths and transitions. Score is calculated based on:
        - Duration satisfaction: how close the adjusted duration is to the target duration.
        - Transition count: fewer transitions yield a higher score.
        - Segment bonus: how many transitions are within the same segment.
        Score is used to evaluate the quality of the arrangement path and filter out less optimal arrangements.

        Args:
            lengths: ArrangementLengths, the lengths of the arrangement including original, adjusted, and target durations.
            transitions: List[Dict[str, Any]], the list of transitions in the arrangement path.
            metadata: Dict[str, Any], metadata containing segments and beats information.

        Returns:
            float: The calculated score for the arrangement path.
        """
        # Score calculation for duration satisfaction
        num_transitions = len(transitions) if transitions else 0
        duration_diff = abs(lengths.adjusted_duration - lengths.target_duration)
        duration_score = max(0, 1.0 - duration_diff / lengths.target_duration)

        # Penalty for big number of transitions (using exponential decay to reward fewer transitions)
        transition_score = 1.0 / (1.0 + num_transitions * 0.2)

        segment_bonus = 0.0

        if transitions:
            same_segment_count = 0

            for trans in transitions:
                from_segment = get_beat_segment(trans['from_beat'], metadata)
                to_segment = get_beat_segment(trans['to_beat'], metadata)

                if from_segment == to_segment:
                    same_segment_count += 1

            segment_bonus = (same_segment_count / len(transitions))

        # TODO? add score for the inclusion of all segments in the arrangement

        return (
                duration_score * self.config.duration_score_weight
                +
                segment_bonus * self.config.segment_bonus_weight
                +
                transition_score * self.config.transition_score_weight
        )

    def find_arrangements(self,
                          target_duration: float,
                          beat_transitions: dict,
                          audio_metadata: Dict[str, Any],
                          k: int = 10,
                          ) -> List[ArrangementPath]:
        """
        Find all possible arrangements of audio segments that meet the target duration using a depth-first search (DFS) algorithm.

        Args:
            k: int, the maximum number of arrangements to return.
            target_duration: float, the target duration for the arrangement in seconds.
            beat_transitions: dict, a dictionary where keys are beats and values are lists of transitions for each beat.
            audio_metadata: Dict[str, Any], metadata containing information about the audio file, including its duration and beats.

        Returns:
            List[ArrangementPath]: A list of ArrangementPath objects representing all valid arrangements found.
        """
        paths = []

        graph = create_beat_graph(beat_transitions)
        transition_beats = list(graph.keys())
        visited = set()

        original_duration = audio_metadata.get("duration")
        start_beat_idx = 0
        start_duration_adjustment = 0
        start_path = []
        start_num_transitions = 0

        logger.info(f"Starting arrangement search for target duration: {target_duration}s "
                    f"with tolerance: {self.config.tolerance}s")

        def dfs(current_beat_idx, duration_adjustment, path, num_transitions):
            total_duration = original_duration + duration_adjustment

            if abs(total_duration - target_duration) <= self.config.tolerance:
                lengths = ArrangementLengths(
                    original_duration=original_duration,
                    adjusted_duration=total_duration,
                    target_duration=target_duration
                )

                score = self.calculate_path_score(lengths, path, audio_metadata)

                paths.append(ArrangementPath(
                    transitions=path.copy(),
                    lengths=lengths,
                    score=score
                ))

            if current_beat_idx >= len(transition_beats):
                return

            state = (current_beat_idx, round(duration_adjustment, 2), num_transitions)

            if state in visited:
                return

            visited.add(state)

            # Check the path where we do not skip or loop the current beat
            if current_beat_idx + 1 < len(transition_beats):
                dfs(current_beat_idx + 1, duration_adjustment, path, num_transitions)

            for trans in graph.get(transition_beats[current_beat_idx], []):
                to_beat = trans['to']
                dur = trans['dur']

                new_adjustment = duration_adjustment - dur

                skip_action = {
                    "type": "skip",
                    "from_beat": transition_beats[current_beat_idx],
                    "to_beat": to_beat,
                    "duration": dur
                }

                if to_beat in transition_beats:
                    to_beat_idx = transition_beats.index(to_beat)
                elif any(beat for beat in transition_beats if beat > to_beat):
                    to_beat_idx = next(i for i, beat in enumerate(transition_beats) if beat > to_beat)
                else:
                    to_beat_idx = len(transition_beats)

                # Skip the current beat and go to the next transition
                dfs(to_beat_idx, new_adjustment, path + [skip_action], num_transitions + 1)

                for n in range(2, self.config.max_loops + 1):
                    added_dur = (n - 1) * dur
                    new_adjustment = duration_adjustment + added_dur
                    loop_action = {
                        "type": "loop",
                        "from_beat": transition_beats[current_beat_idx],
                        "to_beat": to_beat,
                        "duration": dur,
                        "loop_count": n
                    }

                    # Loop the current beat n times
                    dfs(to_beat_idx, new_adjustment, path + [loop_action], num_transitions + 1)

        # Start the DFS from the first beat in the transitions list
        dfs(start_beat_idx, start_duration_adjustment, start_path, start_num_transitions)

        paths.sort(key=lambda x: x.score, reverse=True)

        filtered_paths = [p for p in paths if not has_redundant_chain(p, graph)]

        logger.info(f"Found {len(paths)} arrangements before filtering redundant chains.")
        logger.info(f"Filtered {len(filtered_paths)} arrangements after removing redundant chains.")

        return filtered_paths[:self.config.arrangement_amount]  # Return only the top k arrangements
