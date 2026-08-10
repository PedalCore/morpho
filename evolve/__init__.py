# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evolutionary research layer over the MorphoHDL sequential extension.

The pipeline under test is always:

    genome -> Morpho cell -> compile_seq -> phenotype -> simulate -> fitness

Run Experiment 0 (evolve per-cell rules of a non-uniform CA) from the repo
root with:  python3 -m evolve.experiment0 --task density
"""
