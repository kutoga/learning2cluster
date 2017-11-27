"""

"""
import threading

from ext_clust.common.extrapolation.speaker_factory import create_all_speakers


def setup_suite():
    """
    Can be called whenever the project must be setup on a new machine. It automatically
    generates all not yet generated speaker pickles in the right place.
    """
    for speaker in create_all_speakers():
        if not speaker.is_pickle_saved():
            threading.Thread(target=speaker.safe_to_pickle).start()


def is_suite_setup():
    """
    Checks if all speaker pickles are already generated.
    :return: true if the suite is setup, false otherwise
    """
    for speaker in create_all_speakers():
        if not speaker.is_pickle_saved():
            return False

    return True
