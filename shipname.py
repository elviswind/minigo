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
"""Utilities for converting between representations of model numbers.

i.e. the number 135 corresponds to the string 000135-some-name and so on.
"""


import random
import re
import petname

MODEL_NUM_REGEX = r"^\d{6}"
MODEL_NAME_REGEX = r"^\d{6}(-\w+)+"

def generate(model_num):
    """Generates a new model name, given the model number."""
    if model_num == 0:
        new_name = 'bootstrap'
    else:
        new_name = petname.generate()
    full_name = "%06d-%s" % (model_num, new_name)
    return full_name


def detect_model_num(string):
    """Takes a string related to a model name and extract its model number.

    For example:
        '000000-bootstrap.index' => 0
    """
    match = re.match(MODEL_NUM_REGEX, string)
    if match:
        return int(match.group())
    return None


def detect_model_name(string):
    """Takes a string related to a model name and extract its model name.

    For example:
        '000000-bootstrap.index' => '000000-bootstrap'
    """
    match = re.match(MODEL_NAME_REGEX, string)
    if match:
        return match.group()
    return None