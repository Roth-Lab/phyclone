'''
This file is part of PhyClone.

PhyClone is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PhyClone is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PhyClone.  If not, see
<http://www.gnu.org/licenses/>.

Created on 2013-06-12

@author: Andrew Roth
'''
from math import exp

import os


def make_directory(target_dir):
    '''
    Make target directory if it does not exist.
    '''
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def make_parent_directory(file_name):
    '''
    Given a file name, make the parent directory if it does not exist using make_directory.

    For example, given /some/where/foo.bar make the folder /some/where.
    '''
    parent_dir = os.path.dirname(file_name)

    make_directory(parent_dir)


def normalize(numbers):
    s = sum(numbers)

    return [number / s for number in numbers]


def exp_normalize(numbers):
    m = max(numbers)

    return normalize([exp(number - m) for number in numbers])
