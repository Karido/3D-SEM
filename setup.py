# Copyright (c) 2023, Stefan Toeberg.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# (http://opensource.org/licenses/BSD-3-Clause)
#
# __author__ = "Stefan Toeberg, LUH: IMR"

from setuptools import setup
import os.path


def read(rel_path):
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), rel_path), 'r') as fp:
        return fp.read()


# def read(rel_path):
#     here = os.path.abspath(os.path.dirname(__file__))
#     with codecs.open(os.path.join(here, rel_path), 'r') as fp:
#         return fp.read()


# def get_version(rel_path):
#     for line in read(rel_path).splitlines():
#         if line.startswith('__version__'):
#             delim = '"' if '"' in line else "'"
#             return line.split(delim)[1]
#     else:
#         raise RuntimeError("Unable to find version string.")

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':

    setup(
        name='3D-SEM',
        version=get_version('sem3d/__init__.py'),
        description='3D Reconstruction based on SEM images',
        author='Stefan Toeberg',
        packages=['sem3d'],
        keywords='computer vision, '
                 'scanning electron microscopy, '
                 '3D reconstruction, '
                 'multi view geometry, '
                 'affine camera',
        url='https://github.com/Karido/3D-SEM',
        install_requires=parse_requirements('requirements.txt'),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD-3 License",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        package_data={
            'sem3d': ['dsa/*.png', 'quarz/*.png', 'weights/*.h5']
        },)
