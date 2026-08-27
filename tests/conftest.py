"""Point the whole test run at a scratch data directory.

The storage tests call storage.reset(), which deletes every pothole, every
detection and every evidence JPEG. Run against the default data directory,
`pytest` would therefore destroy a real survey - including the one recorded
on the drive you are about to demo. Setting POTHOLESENSE_DATA_DIR here, before
config.py is imported anywhere, makes that impossible.
"""
import os
import tempfile

os.environ.setdefault(
    "POTHOLESENSE_DATA_DIR",
    tempfile.mkdtemp(prefix="potholesense-test-"),
)
