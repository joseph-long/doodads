import os
import subprocess
import shlex
from .. import utils

def _convert_castelli_kurucz(download_filepath, output_filepath):
    os.makedirs(output_filepath, exist_ok=True)
    subprocess.check_call(shlex.split(f"tar xzf {download_filepath} --strip-components=5 -C {output_filepath}"))

CK_SYNPHOT_ARCHIVE = utils.REMOTE_RESOURCES.add_from_url(
    module=__name__,
    url='https://ssb.stsci.edu/trds/tarfiles/synphot3.tar.gz',
    converter_function=_convert_castelli_kurucz,
    output_filename='ck04models',
)
