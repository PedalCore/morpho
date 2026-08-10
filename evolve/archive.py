# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Run logging: one JSON record per generation, plus best-genome tracking.
Kept append-only so long runs are inspectable while still in flight."""

import json


class RunLog:
    def __init__(self, path=None):
        self.path, self.best, self.best_fit = path, None, float('-inf')
        if path:
            open(path, 'w').close()

    def log(self, record, genome=None, fitness=None):
        if fitness is not None and fitness > self.best_fit:
            self.best, self.best_fit = genome.copy(), fitness
        if self.path:
            with open(self.path, 'a') as f:
                f.write(json.dumps(record) + '\n')
