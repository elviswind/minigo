# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import sys
import time
import numpy as np
from mcts import MCTSNode, MAX_DEPTH

import go

# When to do deterministic move selection.  ~30 moves on a 19x19, ~8 on 9x9
TEMPERATURE_CUTOFF = int(go.M / 4)


def time_recommendation(move_num, seconds_per_move=5, time_limit=15 * 60,
                        decay_factor=0.98):
    '''Given the current move number and the 'desired' seconds per move, return
    how much time should actually be used. This is intended specifically for
    CGOS time controls, which has an absolute 15-minute time limit.

    The strategy is to spend the maximum possible moves using seconds_per_move,
    and then switch to an exponentially decaying time usage, calibrated so that
    we have enough time for an infinite number of moves.'''

    # Divide by two since you only play half the moves in a game.
    player_move_num = move_num / 2

    # Sum of geometric series maxes out at endgame_time seconds.
    endgame_time = seconds_per_move / (1 - decay_factor)

    if endgame_time > time_limit:
        # There is so little main time that we're already in 'endgame' mode.
        base_time = time_limit * (1 - decay_factor)
        core_moves = 0
    else:
        # Leave over endgame_time seconds for the end, and play at
        # seconds_per_move for as long as possible.
        base_time = seconds_per_move
        core_moves = (time_limit - endgame_time) / seconds_per_move

    return base_time * decay_factor ** max(player_move_num - core_moves, 0)


class MCTSPlayerMixin:
    # If `simulations_per_move` is nonzero, it will perform that many reads
    # before playing. Otherwise, it uses `seconds_per_move` of wall time.
    def __init__(self, network, seconds_per_move=5, simulations_per_move=0,
                 verbosity=0, num_parallel=8):
        self.network = network
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.verbosity = verbosity
        self.temp_threshold = TEMPERATURE_CUTOFF
        self.num_parallel = num_parallel
        self.qs = []
        self.comments = []
        self.searches_pi = []
        self.root = None
        super().__init__()

    def initialize_game(self, position=None):
        if position is None:
            position = go.Position()
        self.root = MCTSNode(position)
        self.comments = []
        self.searches_pi = []
        self.qs = []

    def suggest_move(self, position):
        ''' Used for playing a single game.
        For parallel play, use initialize_move, select_leaf,
        incorporate_results, and pick_move
        '''
        start = time.time()

        if self.simulations_per_move == 0:
            while time.time() - start < self.seconds_per_move:
                self.tree_search()
        else:
            current_readouts = self.root.N
            while self.root.N < current_readouts + self.simulations_per_move:
                self.tree_search()
            if self.verbosity > 0:
                print("%d: Searched %d times in %s seconds\n\n" % (
                    position.n, self.simulations_per_move, time.time() - start), file=sys.stderr)

        # print some stats on anything with probability > 1%
        if self.verbosity > 2:
            print(self.root.describe(), file=sys.stderr)
            print('\n\n', file=sys.stderr)
        if self.verbosity > 3:
            print(self.root.position, file=sys.stderr)

        return self.pick_move()

    def play_move(self, c):
        '''
        Notable side effects:
          - finalizes the probability distribution according to
          this roots visit counts into the class' running tally, `searches_pi`
          - Makes the node associated with this move the root, for future
            `inject_noise` calls.
        '''
        self.searches_pi.append(
            self.root.children_as_pi(self.root.position.n < self.temp_threshold))
        self.qs.append(self.root.Q)  # Save our resulting Q.
        self.comments.append(self.root.describe())
        try:
            self.root = self.root.maybe_add_child(c)
        except go.IllegalMove:
            print("Illegal move")
            self.searches_pi.pop()
            self.qs.pop()
            self.comments.pop()
            return False
        self.position = self.root.position  # for showboard
        del self.root.parent.children
        return True  # GTP requires positive result.

    def pick_move(self):
        '''Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max.'''
        if self.root.position.n > self.temp_threshold:
            fcoord = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            fcoord = cdf.searchsorted(selection)
            assert self.root.child_N[fcoord] != 0
        return fcoord

    def tree_search(self, num_parallel=None):
        if num_parallel is None:
            num_parallel = self.num_parallel
        leaves = []
        failsafe = 0
        while len(leaves) < num_parallel and failsafe < num_parallel * 2:
            failsafe += 1
            leaf = self.root.select_leaf()
            if self.verbosity >= 4:
                print(self.show_path_to_root(leaf))
            # if game is over, override the value estimate with the true score
            if leaf.is_done():
                leaf.backup_value(leaf.position.score(), up_to=self.root)
                continue
            leaves.append(leaf)
        if leaves:
            move_probs, values = self.network.run_many(
                [leaf.position for leaf in leaves])
            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.incorporate_results(move_prob, value, up_to=self.root)
        return leaves

    def show_path_to_root(self, node):
        pos = node.position
        diff = node.position.n - self.root.position.n
        if len(pos.recent) == 0:
            return

        path = " ".join(str(move) for move in pos.recent[-diff:])
        if node.position.n >= MAX_DEPTH:
            path += " (depth cutoff reached) %0.1f" % node.position.score()
        elif node.position.is_game_over():
            path += " (game over) %0.1f" % node.position.score()
        return path

    def is_done(self):
        return self.root.is_done()

    def extract_data(self):
        assert len(self.searches_pi) == self.root.position.n
        for pwc, pi in zip(go.replay_position(self.root.position, self.root.position.score()),
                           self.searches_pi):
            yield pwc.position, pi, pwc.result
